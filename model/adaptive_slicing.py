import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveSlicing(nn.Module):
    """
    Adaptive Slicing — Transolver-inspired Physics-Attention for MILP.

    Compresses N variable-node embeddings (z_var_0) into K latent "slice"
    tokens via learnable soft-clustering.  Each slice captures a group of
    variables that share similar solver-state characteristics (e.g. fractional
    LP values, high objective coefficients).

    Architecture (per forward call):
        1. Project z_var_0 → Q  [N, D]      (queries, from variable nodes)
           Project slice_centers → K_c [K, D]  (keys, from learnable centers)
           Project z_var_0 → V  [N, D]      (values, from variable nodes)
        2. Multi-head cross-attention:
              attn = Softmax( Q·K_c^T / sqrt(d_head) / τ )   → [N, H, K]
           where τ is a learnable temperature.
        3. Weighted aggregation (Slicing):
              tokens = Σ_n attn[n]·V[n]  (per graph)          → [B*K, D]
           Normalized by attention-weight sum for size-invariance.
        4. Output projection + LayerNorm.

    Training tricks:
        - Multi-head attention: richer per-slice representations.
        - Learnable temperature (τ): lets the model adaptively sharpen
          or soften the assignment distribution during training.
        - Orthogonal initialization for slice centers: maximizes initial
          diversity so that slices capture distinct variable groups from
          the start.
        - Weighted-average aggregation: divides by Σ_n attn[n,k] per
          slice, making the representation invariant to graph size.
        - Attention dropout: regularizes the soft-clustering.
        - LayerNorm on output: stabilizes downstream processing.
        - Entropy regularization helper: penalizes degenerate attention
          (all variables assigned to one slice).
        - Diversity regularization helper: penalizes slice centers that
          collapse to the same point.
    """

    def __init__(self, emb_size=64, n_slices=64, n_heads=4, dropout=0.1):
        """
        Parameters
        ----------
        emb_size : int
            Hidden dimension D (must be divisible by n_heads).
        n_slices : int
            Number of latent slice tokens K.
        n_heads : int
            Number of attention heads.
        dropout : float
            Dropout rate on attention weights.
        """
        super().__init__()
        assert emb_size % n_heads == 0, \
            f"emb_size ({emb_size}) must be divisible by n_heads ({n_heads})"

        self.emb_size = emb_size
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = emb_size // n_heads

        # ── Learnable slice centers [K, D] ──
        self.slice_centers = nn.Parameter(torch.empty(n_slices, emb_size))

        # ── Learnable temperature (log-scale for positivity) ──
        # Initialized to τ=1.0 (log(1)=0).
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # ── Q / K / V projections ──
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)

        # ── Output projection ──
        self.out_proj = nn.Linear(emb_size, emb_size)

        # ── LayerNorm on output tokens ──
        self.layernorm = nn.LayerNorm(emb_size)

        # ── Attention dropout ──
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self):
        # Orthogonal init for slice centers → maximum initial diversity
        nn.init.orthogonal_(self.slice_centers)
        # Scale so that initial dot-products are O(1)
        with torch.no_grad():
            self.slice_centers.mul_(math.sqrt(self.emb_size / self.n_slices))
        # Xavier for linear projections
        for linear in (self.W_q, self.W_k, self.W_v, self.out_proj):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, z_var_0, batch=None):
        """
        Parameters
        ----------
        z_var_0 : Tensor [N, D]
            Variable-node embeddings from GraphInitialization.
        batch : LongTensor [N], optional
            Graph-membership index for each node (from torch_geometric
            batching).  If None, assumes a single graph.

        Returns
        -------
        latent_tokens : Tensor [B*K, D]
            Aggregated slice tokens (B graphs × K slices).
        token_batch : LongTensor [B*K]
            Graph-membership index for each token.
        attn_weights : Tensor [N, K]
            Head-averaged attention weights (for regularization losses).
        """
        N, D = z_var_0.shape
        H = self.n_heads
        hd = self.head_dim
        K = self.n_slices

        # Determine batch size
        if batch is None:
            batch = z_var_0.new_zeros(N, dtype=torch.long)
        B = batch.max().item() + 1

        # ── Project ──
        Q = self.W_q(z_var_0).view(N, H, hd)           # [N, H, hd]
        K_c = self.W_k(self.slice_centers).view(K, H, hd)  # [K, H, hd]
        V = self.W_v(z_var_0).view(N, H, hd)           # [N, H, hd]

        # ── Attention scores ──
        temperature = self.log_temperature.exp().clamp(min=0.01, max=10.0)
        # [N, H, K] = Q[N,H,hd] · K_c[K,H,hd]^T  (per-head dot product)
        attn_logits = torch.einsum('nhd,khd->nhk', Q, K_c)
        attn_logits = attn_logits / (math.sqrt(hd) * temperature)

        # Softmax over K (which slice does each variable belong to)
        attn = F.softmax(attn_logits, dim=-1)     # [N, H, K]
        attn = self.attn_dropout(attn)

        # ── Weighted aggregation per graph ──
        #    For each graph g, head h, slice k:
        #        numer[g,h,k,:] = Σ_{n ∈ g}  attn[n,h,k] · V[n,h,:]
        #        denom[g,h,k]   = Σ_{n ∈ g}  attn[n,h,k]
        #        tokens[g,k,h,:] = numer / denom     (weighted average)
        #
        #    We flatten (H, K, hd) → H*K*hd so scatter_add runs in one call.

        # Numerator: attn[n,h,k] * V[n,h,d] → [N, H, K, hd]
        weighted_V = attn.unsqueeze(-1) * V.unsqueeze(2)   # [N, H, K, hd]
        weighted_flat = weighted_V.reshape(N, H * K * hd)   # [N, H*K*hd]

        batch_exp = batch.unsqueeze(-1).expand(N, H * K * hd)
        numer = z_var_0.new_zeros(B, H * K * hd)
        numer.scatter_add_(0, batch_exp, weighted_flat)      # [B, H*K*hd]
        numer = numer.view(B, H, K, hd)                      # [B, H, K, hd]

        # Denominator: Σ attn[n,h,k]
        attn_flat = attn.reshape(N, H * K)
        batch_exp_d = batch.unsqueeze(-1).expand(N, H * K)
        denom = z_var_0.new_zeros(B, H * K)
        denom.scatter_add_(0, batch_exp_d, attn_flat)        # [B, H*K]
        denom = denom.view(B, H, K, 1).clamp(min=1e-6)      # [B, H, K, 1]

        # Normalized tokens (weighted average)
        tokens = numer / denom                               # [B, H, K, hd]

        # Reshape: [B, K, H, hd] → [B, K, D] → [B*K, D]
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, K, D)
        tokens = tokens.reshape(B * K, D)

        # ── Output projection + LayerNorm ──
        tokens = self.out_proj(tokens)
        tokens = self.layernorm(tokens)

        # ── Token batch indices ──
        token_batch = torch.arange(B, device=z_var_0.device).unsqueeze(1) \
                           .expand(B, K).reshape(B * K)

        # ── Head-averaged attention weights for regularization ──
        attn_weights = attn.mean(dim=1)  # [N, K]

        return tokens, token_batch, attn_weights

    # ------------------------------------------------------------------
    # Regularization helpers (call during training, add to main loss)
    # ------------------------------------------------------------------
    def entropy_loss(self, attn_weights, batch=None):
        """
        Encourage each variable to spread its attention across slices,
        preventing degenerate collapse where all variables map to one
        slice.

        Returns the *negative* mean entropy (minimizing this → higher
        entropy → more uniform attention).

        Parameters
        ----------
        attn_weights : Tensor [N, K]
            Head-averaged attention weights from forward().
        batch : LongTensor [N], optional
            Graph-membership (unused here, but kept for API consistency).

        Returns
        -------
        loss : scalar Tensor
        """
        # Per-variable entropy: H(p) = -Σ_k p_k log(p_k)
        eps = 1e-8
        ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # [N]
        return -ent.mean()

    def diversity_loss(self):
        """
        Penalize slice centers that are too similar (cosine similarity
        close to 1), encouraging the K slices to cover distinct regions
        of the embedding space.

        Returns
        -------
        loss : scalar Tensor
            Mean off-diagonal cosine similarity (lower is more diverse).
        """
        # Normalize centers
        centers_norm = F.normalize(self.slice_centers, dim=-1)   # [K, D]
        # Cosine similarity matrix [K, K]
        sim = centers_norm @ centers_norm.T
        # Mask diagonal
        K = self.n_slices
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        return sim[mask].mean()
