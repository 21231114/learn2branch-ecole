import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


# ======================================================================
# Building blocks
# ======================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer 2020, used in LLaMA / PaLM).

    Replaces the standard  Linear → ReLU → Linear  with:
        out = W3( SiLU(W1·x) ⊙ W2·x )

    Two parallel projections (W1 gate, W2 value) followed by element-wise
    product and down-projection.  Empirically stronger than ReLU/GELU FFN
    at the same parameter count.

    Note: with ratio=4, the hidden dim is  D * 4 * 2/3 ≈ D * 2.67
    (same param count as standard 4× FFN, since SwiGLU has 3 matrices
    vs 2 for standard FFN — so we use 2/3 of the ratio to compensate).
    """

    def __init__(self, emb_size, ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(emb_size * ratio * 2 / 3)
        # Round to nearest multiple of 8 for hardware efficiency
        hidden = ((hidden + 7) // 8) * 8

        self.w_gate = nn.Linear(emb_size, hidden, bias=False)  # W1
        self.w_value = nn.Linear(emb_size, hidden, bias=False)  # W2
        self.w_out = nn.Linear(hidden, emb_size, bias=False)    # W3
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


class PreLNTransformerLayer(nn.Module):
    """
    Pre-LayerNorm Transformer layer (GPT-2 / modern LLM style).

    Architecture:
        x = x + Dropout( SelfAttn( LN(x) ) )
        x = x + Dropout( SwiGLU_FFN( LN(x) ) )

    Pre-LN is more stable than Post-LN for deep models because the
    residual path carries un-normalized gradients all the way back,
    avoiding vanishing gradients in deep stacks.
    """

    def __init__(self, emb_size, n_heads, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = SwiGLUFFN(emb_size, ratio=ffn_ratio, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # ── Self-attention sub-layer ──
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop1(attn_out)

        # ── FFN sub-layer ──
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# Main module
# ======================================================================

class LatentTrajectoryEvolution(nn.Module):
    """
    Latent Trajectory Evolution — UPT-inspired evolution operator.

    Simulates the iterative optimization process of a Branch & Bound
    solver in latent space.  A shared Transformer Encoder is applied
    T times to K latent slice-tokens, allowing them to exchange
    information (via Self-Attention) and refine their representations
    (via FFN) at each step.

    Architecture:
        For t = 0, 1, …, T−1:
            h = x + StepEmbed[t]                          (inject step info)
            h = TransformerEncoder(h)                     (L layers of Pre-LN Transformer)
            h = FinalNorm(h)
            x = (1 − gₜ)·x + gₜ·h                       (gated residual update)
        Output: x_T = final evolved tokens

    Key design choices:
        1. Weight sharing: The same L-layer Transformer is reused at
           every evolution step.  This is parameter-efficient (like a
           Neural ODE discretization) and naturally models iterative
           refinement — the operator learns "one step of optimization"
           and applies it repeatedly.

        2. Step embedding: A learnable vector per step is added to the
           input, so the shared Transformer can specialize its behavior
           (e.g. coarse exploration early, fine-tuning late).

        3. Gated residual (evolution gate): Instead of replacing tokens
           entirely, each step blends old and new via a learnable gate
           gₜ ∈ (0, 1).  Initialized small (~0.1), this ensures:
           - Early training: gentle updates, stable gradients.
           - Late training: the gate opens up as the Transformer learns
             meaningful updates.
           This is analogous to the Euler step-size in Neural ODE.

        4. Pre-LN Transformer: LayerNorm before each sub-layer, with a
           final LayerNorm after the last layer.  The residual stream
           carries raw gradients, preventing vanishing gradients in the
           depth = L × T effective layers.

        5. SwiGLU FFN: Replaces standard ReLU/GELU FFN with the gated
           variant (SiLU(W1·x) ⊙ W2·x), which consistently outperforms
           standard FFN in modern architectures.

        6. Stochastic depth: During training, entire Transformer layers
           are randomly skipped with probability proportional to their
           depth.  This regularizes the deep model and reduces
           effective computation during training.

        7. Gradient checkpointing: Optional memory optimization that
           trades compute for memory by recomputing activations during
           backward pass.  Essential when T × L is large.

        8. Intermediate state collection: All T+1 states
           [x_0, x_1, …, x_T] are returned, enabling:
           - Deep supervision / auxiliary losses at every step.
           - Exponential moving average of predictions.
           - Analysis of the evolution trajectory.

    No positional encoding is used because the K slice-tokens have no
    natural ordering — they are permutation-equivariant by design.
    """

    def __init__(self, emb_size=64, n_layers=4, n_heads=4,
                 ffn_ratio=4, n_evolve_steps=3, dropout=0.1,
                 stochastic_depth_rate=0.1, use_grad_checkpoint=False):
        """
        Parameters
        ----------
        emb_size : int
            Hidden dimension D (must be divisible by n_heads).
        n_layers : int
            Number of Transformer layers per evolution step.
        n_heads : int
            Number of self-attention heads.
        ffn_ratio : int
            FFN hidden-dim multiplier (adjusted for SwiGLU).
        n_evolve_steps : int
            Number of evolution steps T (Transformer applied T times).
        dropout : float
            Dropout rate for attention and FFN.
        stochastic_depth_rate : float
            Maximum layer-drop probability (linearly increases with
            depth).  Set to 0 to disable.
        use_grad_checkpoint : bool
            If True, use gradient checkpointing to save memory at the
            cost of ~30% more compute during backward.
        """
        super().__init__()
        assert emb_size % n_heads == 0

        self.emb_size = emb_size
        self.n_layers = n_layers
        self.n_evolve_steps = n_evolve_steps
        self.use_grad_checkpoint = use_grad_checkpoint

        # ── Step embeddings [T, D] ──
        # Each evolution step gets a unique additive embedding so the
        # shared Transformer can distinguish "early exploration" from
        # "late refinement".
        self.step_embeddings = nn.Parameter(
            torch.zeros(n_evolve_steps, emb_size)
        )

        # ── Evolution gates (one per step) ──
        # Parameterized as raw logits; gate = sigmoid(logit).
        # Initialized to sigmoid⁻¹(0.1) ≈ −2.2 so early training
        # applies only ~10% of the Transformer's update.
        self.gate_logits = nn.Parameter(
            torch.full((n_evolve_steps,), math.log(0.1 / 0.9))
        )

        # ── Shared Transformer encoder (L layers) ──
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(emb_size, n_heads, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])

        # ── Final LayerNorm (required for Pre-LN architecture) ──
        # In Pre-LN, each sub-layer normalizes its *input*, so the
        # output of the last layer is un-normalized.  This final norm
        # ensures the Transformer's output has stable statistics.
        self.final_norm = nn.LayerNorm(emb_size)

        # ── Stochastic depth drop rates (linear schedule) ──
        # Layer i has drop probability: i/(L-1) * stochastic_depth_rate
        if n_layers > 1:
            self.drop_rates = [
                stochastic_depth_rate * i / (n_layers - 1)
                for i in range(n_layers)
            ]
        else:
            self.drop_rates = [0.0]

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # Step embeddings: small random init (not zero — breaks symmetry
        # between steps while keeping magnitudes small).
        nn.init.normal_(self.step_embeddings, std=0.02)

        # Linear layers: Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _apply_transformer(self, x):
        """
        Run x through the L-layer Transformer with optional stochastic
        depth and gradient checkpointing.

        Parameters
        ----------
        x : Tensor [B, K, D]

        Returns
        -------
        Tensor [B, K, D]
        """
        for i, layer in enumerate(self.layers):
            # ── Stochastic depth ──
            if self.training and self.drop_rates[i] > 0:
                if torch.rand(1).item() < self.drop_rates[i]:
                    continue  # skip this layer entirely

            # ── Gradient checkpointing ──
            if self.use_grad_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.final_norm(x)

    # ------------------------------------------------------------------
    def forward(self, latent_tokens, token_batch=None):
        """
        Parameters
        ----------
        latent_tokens : Tensor [B*K, D]
            Slice tokens from AdaptiveSlicing (Step 2).
        token_batch : LongTensor [B*K], optional
            Graph-membership index for each token.  If None, assumes
            a single graph (B=1).

        Returns
        -------
        evolved_tokens : Tensor [B*K, D]
            Final evolved latent tokens after T steps.
        token_batch : LongTensor [B*K]
            Graph-membership (passed through unchanged).
        intermediate_states : list of Tensor [B*K, D]
            All T+1 states [x_0, x_1, …, x_T] for auxiliary losses.
        """
        total = latent_tokens.shape[0]
        D = self.emb_size

        # ── Reshape [B*K, D] → [B, K, D] ──
        if token_batch is not None:
            B = token_batch.max().item() + 1
            K = total // B
        else:
            B = 1
            K = total
            token_batch = torch.zeros(total, dtype=torch.long,
                                      device=latent_tokens.device)

        x = latent_tokens.view(B, K, D)   # [B, K, D]

        # ── Collect trajectory ──
        intermediate_states = [x.reshape(B * K, D)]

        # ── T evolution steps ──
        for t in range(self.n_evolve_steps):
            # Inject step embedding (broadcast over B and K)
            h = x + self.step_embeddings[t]           # [B, K, D]

            # Apply shared Transformer
            h = self._apply_transformer(h)            # [B, K, D]

            # Gated residual update
            gate = torch.sigmoid(self.gate_logits[t]) # scalar ∈ (0,1)
            x = (1 - gate) * x + gate * h             # [B, K, D]

            intermediate_states.append(x.reshape(B * K, D))

        # ── Reshape back to [B*K, D] ──
        evolved_tokens = x.reshape(B * K, D)

        return evolved_tokens, token_batch, intermediate_states

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def get_gate_values(self):
        """Return current gate values for monitoring during training."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_logits).cpu().tolist()

    def auxiliary_loss(self, intermediate_states, loss_fn, target,
                       ema_decay=0.9):
        """
        Compute weighted auxiliary loss over intermediate evolution
        states.  Later steps receive exponentially higher weight,
        encouraging the model to produce progressively better
        predictions.

        Parameters
        ----------
        intermediate_states : list of Tensor [B*K, D]
            States [x_0, x_1, …, x_T] from forward().
        loss_fn : callable(predictions, target) → scalar
            A function that takes token embeddings and computes a loss
            against the target.  The user defines this based on their
            downstream task (e.g. decode → MSE against optimal solution).
        target : any
            Target data passed to loss_fn.
        ema_decay : float
            Exponential weight decay.  Step t gets weight
            (1-decay) * decay^(T-t), normalized to sum to 1.

        Returns
        -------
        weighted_loss : scalar Tensor
        """
        T = len(intermediate_states) - 1  # T evolution steps
        if T == 0:
            return loss_fn(intermediate_states[0], target)

        # Exponential weights: later steps matter more
        # w_t = decay^(T - t)  for t = 0, 1, ..., T
        weights = torch.tensor(
            [ema_decay ** (T - t) for t in range(T + 1)],
            device=intermediate_states[0].device,
        )
        weights = weights / weights.sum()  # normalize

        total_loss = sum(
            w * loss_fn(state, target)
            for w, state in zip(weights, intermediate_states)
        )
        return total_loss
