"""
Constraint Violation Penalty Loss — soft feasibility enforcement.

During training, penalizes model predictions that violate the MILP
constraints.  This acts as a "boundary potential repulsion" that pushes
the learned solution toward the feasible region.

The penalty is computed in the normalized feature space provided by
ecole's NodeBipartite observation, so it works directly with the
bipartite graph data stored in each sample.

Mathematical formulation
------------------------
Given:
    A : sparse constraint matrix  (from edge_indices + edge_features)
    b : right-hand side vector    (from constraint_features[:, bias_col])
    x : soft prediction vector    (differentiable, from model output)

Violation for constraint i:
    v_i = ReLU( sum_j A[i,j] * x_j  -  b_i )

Total penalty:
    L_violation = mean(v) + alpha * max(v)

The max term ensures that even a single severely violated constraint
is penalized, not hidden by averaging.

How x is constructed (differentiable):
    - Binary vars:       prob_bin (sigmoid output, in [0, 1])
    - Small-range ints:  expected value = sum(k * softmax(logits)[k]) + offset
    - Large-range ints:  raw regression output
    - Continuous vars:   LP relaxation value from features (col 8, fixed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Soft prediction construction
# ======================================================================

def build_soft_predictions(result, n_vars, var_types,
                           variable_features=None, lp_sol_col=8):
    """
    Construct a differentiable prediction vector for all N variables.

    Uses soft (continuous) representations so that gradients can flow
    back through the constraint violation penalty to the model.

    Parameters
    ----------
    result : dict
        Output from DeslicingDecoder.forward().
    n_vars : int
        Total number of variables N.
    var_types : LongTensor [N]
        Per-variable type (0=continuous, 1=binary, 2=integer).
    variable_features : Tensor [N, F], optional
        Raw variable features.  If provided, LP relaxation values
        (column lp_sol_col) are used for continuous variables.
    lp_sol_col : int
        Column index of LP solution value in variable_features.

    Returns
    -------
    x : Tensor [N]
        Differentiable prediction vector.
    """
    # Determine device from any available tensor in result
    for key in ('prob_bin', 'logits_int_small', 'pred_int_large', 'z_out'):
        if key in result and result[key].shape[0] > 0:
            device = result[key].device
            break
    else:
        device = var_types.device

    x = torch.zeros(n_vars, device=device)

    # ---- Binary: use probability directly (differentiable) ----
    if result['idx_bin'].shape[0] > 0:
        x[result['idx_bin']] = result['prob_bin'].squeeze(-1)

    # ---- Small-range integer: expected value (differentiable) ----
    if result['idx_int_small'].shape[0] > 0:
        logits = result['logits_int_small']             # [N_is, C]
        probs = F.softmax(logits, dim=-1)               # [N_is, C]
        C = probs.shape[1]
        class_values = torch.arange(C, device=device, dtype=torch.float)
        expected = (probs * class_values.unsqueeze(0)).sum(dim=-1)
        # Add offset to shift back to original range
        expected = expected + result['int_small_offsets']
        x[result['idx_int_small']] = expected

    # ---- Large-range integer: raw regression output ----
    if result['idx_int_large'].shape[0] > 0:
        x[result['idx_int_large']] = result['pred_int_large'].squeeze(-1)

    # ---- Continuous: LP relaxation (fixed, no gradient) ----
    continuous_mask = (var_types == 0)
    if continuous_mask.any() and variable_features is not None:
        lp_vals = variable_features[continuous_mask, lp_sol_col].detach()
        x[continuous_mask] = lp_vals

    return x


# ======================================================================
# Constraint violation computation
# ======================================================================

def compute_constraint_violation(x, edge_indices, edge_features,
                                 constraint_features, bias_col=1):
    """
    Compute per-constraint violation: v_i = ReLU(A_i · x - b_i).

    Uses the bipartite graph structure (sparse A matrix) and the
    normalized bias from constraint features.

    Parameters
    ----------
    x : Tensor [N_var]
        Soft prediction vector (from build_soft_predictions).
    edge_indices : LongTensor [2, E]
        Bipartite edges: [0] = constraint idx, [1] = variable idx.
    edge_features : Tensor [E, 1] or [E]
        Normalized A matrix coefficients.
    constraint_features : Tensor [N_con, F_con]
        Constraint node features (from ecole NodeBipartite).
    bias_col : int
        Column index of the normalized RHS (bias) in constraint_features.

    Returns
    -------
    violation : Tensor [N_con]
        Per-constraint violation (0 if feasible, positive if violated).
    """
    n_cons = constraint_features.shape[0]
    edge_coefs = edge_features.view(-1)   # [E]

    # Sparse matrix-vector product: Ax[i] = sum_j A[i,j] * x[j]
    weighted = edge_coefs * x[edge_indices[1]]
    Ax = x.new_zeros(n_cons)
    Ax.scatter_add_(0, edge_indices[0], weighted)

    # RHS
    b = constraint_features[:, bias_col]

    # Violation = ReLU(Ax - b)
    violation = F.relu(Ax - b)
    return violation


# ======================================================================
# Main loss function
# ======================================================================

class ConstraintViolationLoss(nn.Module):
    """
    Soft constraint violation penalty for training.

    Adds a differentiable penalty term to the main task loss that
    discourages predictions from violating MILP constraints.

    Loss = lambda_mean * mean(violation)
         + lambda_max  * max(violation)

    The mean term provides a smooth gradient signal from all violated
    constraints.  The max term ensures that no single constraint is
    catastrophically violated.

    Usage
    -----
    >>> cv_loss = ConstraintViolationLoss(lambda_mean=1.0, lambda_max=0.1)
    >>> result = decoder(...)
    >>> x = build_soft_predictions(result, n_vars, var_types, var_feats)
    >>> loss_cv = cv_loss(x, edge_indices, edge_features, con_feats)
    >>> total_loss = task_loss + loss_cv
    """

    def __init__(self, lambda_mean=1.0, lambda_max=0.1,
                 bias_col=1, lp_sol_col=8):
        """
        Parameters
        ----------
        lambda_mean : float
            Weight for mean violation penalty.
        lambda_max : float
            Weight for max violation penalty (worst-case).
        bias_col : int
            Column of normalized RHS in constraint_features.
        lp_sol_col : int
            Column of LP solution in variable_features.
        """
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_max = lambda_max
        self.bias_col = bias_col
        self.lp_sol_col = lp_sol_col

    def forward(self, result, n_vars, var_types,
                edge_indices, edge_features, constraint_features,
                variable_features=None):
        """
        Compute constraint violation penalty.

        Parameters
        ----------
        result : dict
            Output from DeslicingDecoder.forward().
        n_vars : int
            Total number of variables.
        var_types : LongTensor [N]
            Variable types.
        edge_indices : LongTensor [2, E]
            Bipartite edge indices.
        edge_features : Tensor [E, 1]
            Edge features (A coefficients).
        constraint_features : Tensor [N_con, F_con]
            Constraint features.
        variable_features : Tensor [N, F], optional
            Variable features (for LP relaxation of continuous vars).

        Returns
        -------
        loss_dict : dict
            'penalty'  : scalar — total violation penalty
            'mean_viol': scalar — mean violation across constraints
            'max_viol' : scalar — maximum constraint violation
            'n_violated': int  — number of violated constraints
        """
        # Build differentiable prediction vector
        x = build_soft_predictions(
            result, n_vars, var_types,
            variable_features=variable_features,
            lp_sol_col=self.lp_sol_col,
        )

        # Compute violations
        violation = compute_constraint_violation(
            x, edge_indices, edge_features,
            constraint_features, bias_col=self.bias_col,
        )

        mean_viol = violation.mean()
        max_viol = violation.max() if violation.shape[0] > 0 else x.new_tensor(0.0)
        n_violated = (violation > 1e-6).sum().item()

        penalty = self.lambda_mean * mean_viol + self.lambda_max * max_viol

        return {
            'penalty': penalty,
            'mean_viol': mean_viol,
            'max_viol': max_viol,
            'n_violated': n_violated,
        }
