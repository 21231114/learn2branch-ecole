import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# Constants
# ======================================================================

INF_THRESHOLD = 1e18  # absolute bound values beyond this are "unbounded"

# Column indices for global bounds in variable_features
# (ecole NodeBipartite 19 cols + lb_local, ub_local, lb_global, ub_global)
LB_GLOBAL_COL = 21
UB_GLOBAL_COL = 22


# ======================================================================
# Utility: extract variable types from raw features
# ======================================================================

def extract_var_types(variable_features):
    """
    Extract variable type labels from raw ecole NodeBipartite features.

    Ecole's NodeBipartite variable features encode type as one-hot in
    columns 1-4:
        col 1: is_type_binary
        col 2: is_type_integer
        col 3: is_type_implicit_integer
        col 4: is_type_continuous

    We map these to a single integer label:
        0 = continuous  (col 4, or none of 1-3)
        1 = binary      (col 1)
        2 = integer     (col 2 or col 3, i.e. general integer)

    Parameters
    ----------
    variable_features : Tensor [N, F]  (F >= 5)
        Raw variable features (before embedding).

    Returns
    -------
    var_types : LongTensor [N]
        Per-variable type label: 0=continuous, 1=binary, 2=integer.
    """
    is_binary = variable_features[:, 1] > 0.5
    is_integer = (variable_features[:, 2] > 0.5) | (variable_features[:, 3] > 0.5)

    var_types = torch.zeros(variable_features.shape[0], dtype=torch.long,
                            device=variable_features.device)
    var_types[is_binary] = 1
    var_types[is_integer] = 2
    return var_types


# ======================================================================
# Utility: extract integer routing info (small-range vs large-range)
# ======================================================================

def extract_int_routing(variable_features, var_types, threshold=10,
                        lb_col=LB_GLOBAL_COL, ub_col=UB_GLOBAL_COL):
    """
    Split integer variables (var_types == 2) into small-range and
    large-range based on their global bounds.

    Small-range: both bounds finite AND (ub - lb) <= threshold
        -> decoded with Cross-Entropy over {0, ..., range} classes
    Large-range: any bound infinite OR (ub - lb) > threshold
        -> decoded with Huber regression (single scalar)

    Parameters
    ----------
    variable_features : Tensor [N, F]
        Raw variable features containing bound columns.
    var_types : LongTensor [N]
        Per-variable type label (0=continuous, 1=binary, 2=integer).
    threshold : int
        Maximum range (inclusive) for small-range classification.
        E.g. threshold=10 means ub - lb <= 10 is small-range.
    lb_col : int
        Column index for global lower bound in variable_features.
    ub_col : int
        Column index for global upper bound in variable_features.

    Returns
    -------
    mask_int_small : BoolTensor [N]
        Which variables are small-range integers.
    mask_int_large : BoolTensor [N]
        Which variables are large-range integers.
    int_small_offsets : FloatTensor [N_int_small]
        Lower bounds (floored) for shifting CE targets to [0, range].
    int_small_ranges : LongTensor [N_int_small]
        Number of valid classes per small-range variable (ub - lb + 1).
    """
    device = variable_features.device
    is_int = (var_types == 2)

    lb = variable_features[:, lb_col]
    ub = variable_features[:, ub_col]

    # Finite means both bounds are within reasonable range
    is_finite = (lb.abs() < INF_THRESHOLD) & (ub.abs() < INF_THRESHOLD)
    var_range = ub - lb

    # Small range: integer AND finite bounds AND range <= threshold
    mask_int_small = is_int & is_finite & (var_range <= threshold)
    mask_int_large = is_int & ~mask_int_small

    # Offsets and ranges for small-range CE
    if mask_int_small.any():
        int_small_offsets = lb[mask_int_small].floor()
        int_small_ranges = (ub[mask_int_small].ceil()
                            - int_small_offsets + 1).long()
        # Clamp ranges to [1, threshold + 1] for safety
        int_small_ranges = int_small_ranges.clamp(1, threshold + 1)
    else:
        int_small_offsets = torch.zeros(0, device=device)
        int_small_ranges = torch.zeros(0, dtype=torch.long, device=device)

    return mask_int_small, mask_int_large, int_small_offsets, int_small_ranges


# ======================================================================
# Decoder heads
# ======================================================================

class MLPDecoderHead(nn.Module):
    """
    Multi-layer MLP decoder with LayerNorm and residual connection.

    Architecture:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual -> head
    """

    def __init__(self, emb_size, out_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(emb_size, out_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor [M, D]

        Returns
        -------
        Tensor [M, out_dim]
        """
        h = self.norm(x)
        h = x + self.mlp(h)    # residual
        return self.head(h)


# ======================================================================
# Main module
# ======================================================================

class DeslicingDecoder(nn.Module):
    """
    Deslicing & Masked Decoding with Dynamic Routing (Step 4).

    Maps K evolved latent tokens back to N per-variable predictions,
    then routes discrete variables to type-specific decoder heads
    based on their characteristics.

    Variable routing:
        - Binary variables:           Focal BCE loss
        - Small-range integers (<=T): Cross-Entropy loss (classification)
        - Large/unbounded integers:   Huber loss (regression)

    The routing threshold T (default 10) determines the boundary:
        range = ub_global - lb_global
        if range <= T  ->  small-range CE with (T+1) classes
        if range > T or unbounded  ->  regression with Huber loss

    Architecture:
        1. Deslice: z_out = attn_weights @ evolved_tokens -> [N, D]
        2. Residual fusion: z_out = LayerNorm(z_desliced + z_var_0)
        3. Route: separate bin / int_small / int_large / continuous
        4. Decode:
           - Binary head:     MLP -> Sigmoid    -> [N_bin, 1]
           - Int-small head:  MLP -> masked logits -> [N_int_s, T+1]
           - Int-large head:  MLP -> scalar     -> [N_int_l, 1]

    Training tricks:
        - Dynamic logit masking: For small-range CE, invalid classes
          (beyond the variable's actual range) are masked to -inf before
          softmax. This focuses probability mass on valid values only.
        - Target offsetting: Small-range integer targets are shifted by
          subtracting the lower bound, so CE targets are always in
          [0, range]. At prediction time, the offset is added back.
        - Huber loss: Smooth L1 for large-range regression. Less
          sensitive to outliers than MSE, more stable training.
        - Rounding regularization: Soft penalty encouraging large-range
          predictions to be close to integers: (pred - round(pred))^2.

    Loss formula:
        L = w_bin * FocalBCE(Binary)
          + w_int_s * CE(Small_Int)
          + w_int_l * Huber(Large_Int)
          + w_round * RoundingReg(Large_Int)
    """

    def __init__(self, emb_size=64, int_range_threshold=10, dropout=0.1):
        """
        Parameters
        ----------
        emb_size : int
            Hidden dimension D.
        int_range_threshold : int
            Maximum range (ub - lb) for small-range integer classification.
            Variables with range <= threshold use CE with (threshold+1) classes.
            Variables with range > threshold or unbounded use Huber regression.
        dropout : float
            Dropout rate in decoder heads.
        """
        super().__init__()
        self.emb_size = emb_size
        self.int_range_threshold = int_range_threshold
        self.int_small_classes = int_range_threshold + 1

        # ---- Deslicing projection ----
        self.deslice_proj = nn.Linear(emb_size, emb_size)

        # ---- Fusion LayerNorm ----
        self.fusion_norm = nn.LayerNorm(emb_size)

        # ---- Binary decoder head ----
        # Output: probability of being 1 (Sigmoid applied outside)
        self.bin_decoder = MLPDecoderHead(emb_size, out_dim=1,
                                          dropout=dropout)

        # ---- Small-range integer decoder head (Cross-Entropy) ----
        # Output: logits over {0, 1, ..., threshold}
        self.int_small_decoder = MLPDecoderHead(
            emb_size, out_dim=self.int_small_classes, dropout=dropout)

        # ---- Large-range integer decoder head (Huber regression) ----
        # Output: a single continuous scalar
        self.int_large_decoder = MLPDecoderHead(emb_size, out_dim=1,
                                                dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Binary decoder output bias: slight negative prior
        # (most binary variables are 0 in MILP)
        with torch.no_grad():
            self.bin_decoder.head.bias.fill_(-0.5)

    def forward(self, evolved_tokens, token_batch, attn_weights,
                var_types, z_var_0=None, var_batch=None,
                variable_features=None):
        """
        Parameters
        ----------
        evolved_tokens : Tensor [B*K, D]
            Evolved latent tokens from Step 3.
        token_batch : LongTensor [B*K]
            Graph-membership for each token.
        attn_weights : Tensor [N, K]
            Attention weights from Step 2 (soft-clustering assignment).
        var_types : LongTensor [N]
            Per-variable type label: 0=continuous, 1=binary, 2=integer.
        z_var_0 : Tensor [N, D], optional
            Original variable embeddings from Step 1 (for residual).
        var_batch : LongTensor [N], optional
            Graph-membership for each variable node.
        variable_features : Tensor [N, F], optional
            Raw variable features (for extracting bounds to route
            integers).  If None, all integers default to small-range.

        Returns
        -------
        result : dict with keys:
            'z_out'             : Tensor [N, D]
            'prob_bin'          : Tensor [N_bin, 1]
            'logits_int_small'  : Tensor [N_int_s, T+1]  (masked logits)
            'pred_int_large'    : Tensor [N_int_l, 1]    (raw scalars)
            'mask_bin'          : BoolTensor [N]
            'mask_int'          : BoolTensor [N]  (all integers)
            'mask_int_small'    : BoolTensor [N]
            'mask_int_large'    : BoolTensor [N]
            'idx_bin'           : LongTensor [N_bin]
            'idx_int_small'     : LongTensor [N_int_s]
            'idx_int_large'     : LongTensor [N_int_l]
            'int_small_offsets' : FloatTensor [N_int_s]  (lb offsets)
            'int_small_ranges'  : LongTensor [N_int_s]   (valid classes)
        """
        N = attn_weights.shape[0]
        K = attn_weights.shape[1]
        D = self.emb_size
        device = evolved_tokens.device

        # Determine batch info
        if var_batch is None:
            var_batch = torch.zeros(N, dtype=torch.long, device=device)
        B = var_batch.max().item() + 1

        # ---- Step 1: Deslice ----
        projected_tokens = self.deslice_proj(evolved_tokens)  # [B*K, D]
        tokens_3d = projected_tokens.view(B, K, D)
        tokens_per_var = tokens_3d[var_batch]                 # [N, K, D]
        z_desliced = (attn_weights.unsqueeze(-1) * tokens_per_var).sum(dim=1)

        # ---- Step 2: Residual fusion ----
        if z_var_0 is not None:
            z_out = self.fusion_norm(z_desliced + z_var_0)
        else:
            z_out = self.fusion_norm(z_desliced)

        # ---- Step 3: Route by variable type ----
        mask_bin = (var_types == 1)

        # Integer routing: split into small-range and large-range
        if variable_features is not None:
            (mask_int_small, mask_int_large,
             int_small_offsets, int_small_ranges) = extract_int_routing(
                variable_features, var_types,
                threshold=self.int_range_threshold)
        else:
            # Fallback: all integers treated as small-range
            mask_int_small = (var_types == 2)
            mask_int_large = torch.zeros(N, dtype=torch.bool, device=device)
            n_is = mask_int_small.sum().item()
            int_small_offsets = torch.zeros(n_is, device=device)
            int_small_ranges = torch.full(
                (n_is,), self.int_small_classes, dtype=torch.long,
                device=device)

        mask_int = mask_int_small | mask_int_large

        idx_bin = torch.where(mask_bin)[0]
        idx_int_small = torch.where(mask_int_small)[0]
        idx_int_large = torch.where(mask_int_large)[0]

        z_out_bin = z_out[mask_bin]               # [N_bin, D]
        z_out_int_small = z_out[mask_int_small]   # [N_int_s, D]
        z_out_int_large = z_out[mask_int_large]   # [N_int_l, D]

        # ---- Step 4: Decode ----

        # Binary: MLP -> sigmoid
        if z_out_bin.shape[0] > 0:
            prob_bin = torch.sigmoid(self.bin_decoder(z_out_bin))
        else:
            prob_bin = z_out.new_empty(0, 1)

        # Small-range integer: MLP -> masked logits
        if z_out_int_small.shape[0] > 0:
            logits_int_small = self.int_small_decoder(z_out_int_small)
            # Dynamic masking: set invalid class positions to -inf
            # For variable i with range r_i, only logits[0:r_i] are valid
            range_idx = torch.arange(
                self.int_small_classes, device=device)  # [C]
            valid_mask = (
                range_idx.unsqueeze(0) < int_small_ranges.unsqueeze(1)
            )  # [N_int_s, C]
            logits_int_small = logits_int_small.masked_fill(
                ~valid_mask, -1e9)
        else:
            logits_int_small = z_out.new_empty(0, self.int_small_classes)

        # Large-range integer: MLP -> scalar
        if z_out_int_large.shape[0] > 0:
            pred_int_large = self.int_large_decoder(z_out_int_large)
        else:
            pred_int_large = z_out.new_empty(0, 1)

        return {
            'z_out': z_out,
            'prob_bin': prob_bin,
            'logits_int_small': logits_int_small,
            'pred_int_large': pred_int_large,
            'mask_bin': mask_bin,
            'mask_int': mask_int,
            'mask_int_small': mask_int_small,
            'mask_int_large': mask_int_large,
            'idx_bin': idx_bin,
            'idx_int_small': idx_int_small,
            'idx_int_large': idx_int_large,
            'int_small_offsets': int_small_offsets,
            'int_small_ranges': int_small_ranges,
        }

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def binary_focal_loss(self, prob_bin, target_bin, gamma=2.0,
                          label_smoothing=0.01):
        """
        Focal loss for binary variable prediction.

        Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Parameters
        ----------
        prob_bin : Tensor [N_bin, 1]
        target_bin : Tensor [N_bin] or [N_bin, 1]
        gamma : float
            Focusing parameter. gamma=0 -> standard BCE.
        label_smoothing : float
            Smooth targets: 0 -> eps, 1 -> 1-eps.

        Returns
        -------
        loss : scalar Tensor
        """
        if prob_bin.shape[0] == 0:
            return prob_bin.new_tensor(0.0)

        target = target_bin.view(-1, 1).float()

        if label_smoothing > 0:
            target = target * (1 - label_smoothing) + 0.5 * label_smoothing

        p = prob_bin.clamp(1e-7, 1 - 1e-7)
        p_t = target * p + (1 - target) * (1 - p)
        focal_weight = (1 - p_t) ** gamma
        bce = -(target * p.log() + (1 - target) * (1 - p).log())

        return (focal_weight * bce).mean()

    def integer_ce_loss(self, logits_int_small, target_int_small,
                        offsets, ranges, label_smoothing=0.05):
        """
        Cross-entropy loss for small-range integer variables.

        Targets are shifted by subtracting the lower bound (offset)
        so that CE targets are in [0, range].

        Uses a custom label-smoothing implementation that distributes
        the smoothing mass only over *valid* classes (within each
        variable's actual range), not the masked-out positions.

        Parameters
        ----------
        logits_int_small : Tensor [N_int_s, C]
            Masked logits from forward() (invalid classes at -inf).
        target_int_small : FloatTensor [N_int_s]
            Ground-truth integer values (raw, before shifting).
        offsets : FloatTensor [N_int_s]
            Lower bounds to subtract from targets.
        ranges : LongTensor [N_int_s]
            Number of valid classes per variable (ub - lb + 1).
        label_smoothing : float
            Label smoothing for CE (default 0.05).  Smoothing mass
            is distributed only over valid classes.

        Returns
        -------
        loss : scalar Tensor
        """
        if logits_int_small.shape[0] == 0:
            return logits_int_small.new_tensor(0.0)

        # Shift targets: target_shifted = target - offset
        shifted = (target_int_small.float() - offsets).round().long()
        shifted = shifted.clamp(0, self.int_small_classes - 1)

        # log-softmax (masked logits -> masked classes get ~0 probability)
        log_probs = F.log_softmax(logits_int_small, dim=-1)  # [N, C]

        C = self.int_small_classes
        device = logits_int_small.device

        # Valid-class mask for label smoothing
        range_idx = torch.arange(C, device=device)
        valid_mask = range_idx.unsqueeze(0) < ranges.unsqueeze(1)  # [N, C]

        # Build target distribution
        # One-hot for the true class
        target_dist = torch.zeros_like(log_probs)
        target_dist.scatter_(1, shifted.unsqueeze(1), 1.0)

        if label_smoothing > 0:
            # Distribute smoothing only over valid classes
            n_valid = valid_mask.float().sum(dim=1, keepdim=True)  # [N, 1]
            smooth_per_class = label_smoothing / n_valid.clamp(min=1)
            target_dist = (target_dist * (1 - label_smoothing)
                           + smooth_per_class * valid_mask.float())

        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        return loss

    def integer_huber_loss(self, pred_int_large, target_int_large,
                           delta=1.0):
        """
        Huber (smooth L1) loss for large-range / unbounded integer
        variables.

        Parameters
        ----------
        pred_int_large : Tensor [N_int_l, 1]
            Raw scalar predictions from forward().
        target_int_large : FloatTensor [N_int_l]
            Ground-truth integer values.
        delta : float
            Huber loss threshold. Below delta: quadratic.
            Above delta: linear.

        Returns
        -------
        loss : scalar Tensor
        """
        if pred_int_large.shape[0] == 0:
            return pred_int_large.new_tensor(0.0)

        pred = pred_int_large.squeeze(-1)
        target = target_int_large.float()
        return F.huber_loss(pred, target, delta=delta)

    def integer_rounding_loss(self, pred_int_large):
        """
        Soft rounding regularization: encourages large-range integer
        predictions to be close to the nearest integer.

        Loss = mean( (pred - round(pred))^2 )

        The round() is detached so gradients flow only through pred.

        Parameters
        ----------
        pred_int_large : Tensor [N_int_l, 1]

        Returns
        -------
        loss : scalar Tensor
        """
        if pred_int_large.shape[0] == 0:
            return pred_int_large.new_tensor(0.0)

        pred = pred_int_large.squeeze(-1)
        frac = pred - pred.round().detach()
        return (frac ** 2).mean()

    def combined_loss(self, result, sol_values, var_types,
                      gamma=2.0, label_smoothing_bin=0.01,
                      label_smoothing_int=0.05,
                      huber_delta=1.0,
                      weight_bin=1.0, weight_int_small=1.0,
                      weight_int_large=1.0, weight_round=0.1):
        """
        Combined loss over all variable types with dynamic routing.

        Loss = w_bin * FocalBCE(Binary)
             + w_int_s * CE(Small_Int)
             + w_int_l * Huber(Large_Int)
             + w_round * RoundingReg(Large_Int)

        Parameters
        ----------
        result : dict
            Output from forward().
        sol_values : Tensor [N]
            Ground-truth solution values for all variables.
        var_types : LongTensor [N]
            Variable types (same as passed to forward()).
        gamma : float
            Focal loss gamma for binary variables.
        label_smoothing_bin : float
            Label smoothing for binary focal loss.
        label_smoothing_int : float
            Label smoothing for small-range integer CE.
        huber_delta : float
            Huber loss delta for large-range integers.
        weight_bin : float
            Weight for binary loss.
        weight_int_small : float
            Weight for small-range integer CE loss.
        weight_int_large : float
            Weight for large-range integer Huber loss.
        weight_round : float
            Weight for rounding regularization on large-range preds.

        Returns
        -------
        loss_dict : dict with keys:
            'total'     : scalar
            'binary'    : scalar  (focal BCE)
            'int_small' : scalar  (cross-entropy)
            'int_large' : scalar  (Huber)
            'rounding'  : scalar  (rounding regularization)
        """
        zero = sol_values.new_tensor(0.0)

        # ---- Binary focal loss ----
        if result['mask_bin'].any():
            loss_bin = self.binary_focal_loss(
                result['prob_bin'],
                sol_values[result['mask_bin']],
                gamma=gamma,
                label_smoothing=label_smoothing_bin,
            )
        else:
            loss_bin = zero

        # ---- Small-range integer CE loss ----
        if result['mask_int_small'].any():
            loss_int_small = self.integer_ce_loss(
                result['logits_int_small'],
                sol_values[result['mask_int_small']],
                result['int_small_offsets'],
                result['int_small_ranges'],
                label_smoothing=label_smoothing_int,
            )
        else:
            loss_int_small = zero

        # ---- Large-range integer Huber loss ----
        if result['mask_int_large'].any():
            loss_int_large = self.integer_huber_loss(
                result['pred_int_large'],
                sol_values[result['mask_int_large']],
                delta=huber_delta,
            )
            loss_round = self.integer_rounding_loss(
                result['pred_int_large'],
            )
        else:
            loss_int_large = zero
            loss_round = zero

        total = (weight_bin * loss_bin
                 + weight_int_small * loss_int_small
                 + weight_int_large * loss_int_large
                 + weight_round * loss_round)

        return {
            'total': total,
            'binary': loss_bin,
            'int_small': loss_int_small,
            'int_large': loss_int_large,
            'rounding': loss_round,
        }

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_full(self, result, n_vars):
        """
        Reconstruct full prediction vector for all N variables.

        Continuous variables get NaN (not predicted).
        Binary variables get their predicted probability.
        Small-range integers get argmax + offset (shifted back).
        Large-range integers get rounded regression output.

        Parameters
        ----------
        result : dict
            Output from forward().
        n_vars : int
            Total number of variables N.

        Returns
        -------
        predictions : Tensor [N]
        """
        device = (result['prob_bin'].device
                  if result['prob_bin'].shape[0] > 0
                  else result['logits_int_small'].device
                  if result['logits_int_small'].shape[0] > 0
                  else result['pred_int_large'].device)

        predictions = torch.full((n_vars,), float('nan'), device=device)

        # Binary: probability
        if result['idx_bin'].shape[0] > 0:
            predictions[result['idx_bin']] = (
                result['prob_bin'].squeeze(-1))

        # Small-range integer: argmax + offset
        if result['idx_int_small'].shape[0] > 0:
            argmax_vals = result['logits_int_small'].argmax(dim=-1).float()
            predictions[result['idx_int_small']] = (
                argmax_vals + result['int_small_offsets'])

        # Large-range integer: rounded regression
        if result['idx_int_large'].shape[0] > 0:
            predictions[result['idx_int_large']] = (
                result['pred_int_large'].squeeze(-1).round())

        return predictions
