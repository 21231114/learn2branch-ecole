"""
Trust Region Solver Handoff — inference-time integration with MILP solvers.

Takes trained model predictions and converts them into solver-usable
heuristic information via two complementary strategies:

1. **Trust Region Fixing**: Fix high-confidence discrete variables
   (tighten LB=UB), leaving uncertain variables for the solver.
   With most combinatorial variables "flattened", the solver sees
   a much smaller effective Branch & Bound tree.

2. **MIP Start Injection**: Feed the full predicted solution as a
   warm-start hint.  The solver will attempt to repair it if
   infeasible, using its built-in Sub-MIP machinery.

Backtracking mechanism:
   If fixing leads to infeasibility (detected in Presolve), the
   module automatically reduces the number of fixings by halving
   the confidence threshold and retrying.  This ensures robustness
   without sacrificing speed.

Usage
-----
>>> solver = TrustRegionSolver(threshold_high=0.95, threshold_low=0.05)
>>> fixings = solver.extract_fixings(result, n_vars, var_types)
>>> sol, status, obj = solver.backtracking_solve(
...     'instance.lp', result, n_vars, var_types, var_names)
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict


class TrustRegionSolver:
    """
    Trust Region solver handoff for inference.

    Extracts high-confidence variable fixings from model predictions
    and hands them to a MILP solver (Gurobi or SCIP) with automatic
    backtracking on infeasibility.
    """

    def __init__(self, threshold_high=0.95, threshold_low=0.05,
                 int_confidence=0.90, max_backtrack_steps=4,
                 solver='gurobi', time_limit=60, threads=1, verbose=False):
        """
        Parameters
        ----------
        threshold_high : float
            Binary variables with prob > threshold_high are fixed to 1.
        threshold_low : float
            Binary variables with prob < threshold_low are fixed to 0.
        int_confidence : float
            Small-range integer variables with max softmax probability
            exceeding this threshold are fixed to the argmax value.
        max_backtrack_steps : int
            Maximum number of backtracking attempts before giving up
            fixings entirely.
        solver : str
            Solver backend: 'gurobi' or 'scip'.
        time_limit : float
            Time limit (seconds) for the solver after fixing.
        threads : int
            Number of solver threads.
        verbose : bool
            If True, print solver output.
        """
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.int_confidence = int_confidence
        self.max_backtrack_steps = max_backtrack_steps
        self.solver = solver
        self.time_limit = time_limit
        self.threads = threads
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fixing extraction
    # ------------------------------------------------------------------

    def extract_fixings(self, result, n_vars, var_types,
                        threshold_high=None, threshold_low=None,
                        int_confidence=None):
        """
        Extract confident variable fixings from model predictions.

        Returns an OrderedDict sorted by confidence (highest first),
        enabling easy truncation for backtracking.

        Parameters
        ----------
        result : dict
            Output from DeslicingDecoder.forward().
        n_vars : int
            Total number of variables.
        var_types : LongTensor [N]
            Variable types (0=cont, 1=bin, 2=int).
        threshold_high : float, optional
            Override self.threshold_high.
        threshold_low : float, optional
            Override self.threshold_low.
        int_confidence : float, optional
            Override self.int_confidence.

        Returns
        -------
        fixings : OrderedDict
            Maps variable index (int) -> (fixed_value, confidence).
            Sorted by descending confidence.
        """
        th = threshold_high or self.threshold_high
        tl = threshold_low or self.threshold_low
        ic = int_confidence or self.int_confidence

        candidates = []  # list of (var_idx, fixed_value, confidence)

        # ---- Binary variables ----
        if result['idx_bin'].shape[0] > 0:
            probs = result['prob_bin'].detach().squeeze(-1)
            for i, idx in enumerate(result['idx_bin']):
                p = probs[i].item()
                if p > th:
                    candidates.append((idx.item(), 1, p))
                elif p < tl:
                    candidates.append((idx.item(), 0, 1.0 - p))

        # ---- Small-range integer variables ----
        if result['idx_int_small'].shape[0] > 0:
            logits = result['logits_int_small'].detach()
            softmax_probs = F.softmax(logits, dim=-1)
            max_probs, max_idx = softmax_probs.max(dim=-1)
            offsets = result['int_small_offsets'].detach()
            for i, idx in enumerate(result['idx_int_small']):
                conf = max_probs[i].item()
                if conf > ic:
                    val = (max_idx[i] + offsets[i]).item()
                    candidates.append((idx.item(), int(round(val)), conf))

        # ---- Large-range integer variables ----
        if result['idx_int_large'].shape[0] > 0:
            preds = result['pred_int_large'].detach().squeeze(-1)
            for i, idx in enumerate(result['idx_int_large']):
                val = preds[i].item()
                rounded = round(val)
                frac_dist = abs(val - rounded)
                # Confidence: how close the prediction is to an integer
                # frac_dist near 0 → high confidence
                conf = 1.0 - 2 * min(frac_dist, 0.5)
                if conf > ic:
                    candidates.append((idx.item(), int(rounded), conf))

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return OrderedDict(
            (idx, (val, conf)) for idx, val, conf in candidates
        )

    def get_full_prediction(self, result, n_vars):
        """
        Construct full prediction vector for MIP Start.

        Parameters
        ----------
        result : dict
            Output from DeslicingDecoder.forward().
        n_vars : int
            Total number of variables.

        Returns
        -------
        predictions : dict
            Maps variable index (int) -> predicted value.
        """
        from model.deslicing_decoder import DeslicingDecoder
        decoder = DeslicingDecoder.__new__(DeslicingDecoder)
        preds = DeslicingDecoder.predict_full(decoder, result, n_vars)

        prediction_dict = {}
        for i in range(n_vars):
            v = preds[i].item()
            if not (v != v):  # not NaN
                prediction_dict[i] = v
        return prediction_dict

    # ------------------------------------------------------------------
    # Gurobi solver
    # ------------------------------------------------------------------

    def _solve_gurobi(self, instance_path, fixings_dict, var_names=None,
                      mip_start_dict=None):
        """
        Solve a MILP instance with Gurobi, applying variable fixings.

        Parameters
        ----------
        instance_path : str
            Path to .lp or .mps instance file.
        fixings_dict : dict
            Maps variable index -> fixed value.
        var_names : list of str, optional
            Variable names corresponding to indices.  If None,
            Gurobi variable order is used.
        mip_start_dict : dict, optional
            Maps variable index -> start value (MIP Start).

        Returns
        -------
        solution : dict or None
            {var_name: value} if feasible, None if infeasible.
        status : str
            'optimal', 'feasible', 'infeasible', 'timelimit', etc.
        obj_val : float or None
        info : dict
            Additional info (solving_time, n_nodes, n_fixed).
        """
        import gurobipy as gp
        from gurobipy import GRB

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if self.verbose else 0)
        env.start()
        model = gp.read(instance_path, env)
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('Threads', self.threads)

        grb_vars = model.getVars()

        # Build name -> var mapping if var_names provided
        if var_names is not None:
            # Map var_names[index] to Gurobi variable
            # Handle SCIP transformed name prefix "t_"
            grb_var_by_name = {v.VarName: v for v in grb_vars}
            index_to_grb = {}
            for idx, name in enumerate(var_names):
                if name in grb_var_by_name:
                    index_to_grb[idx] = grb_var_by_name[name]
                elif name.startswith('t_'):
                    orig = name[2:]
                    if orig in grb_var_by_name:
                        index_to_grb[idx] = grb_var_by_name[orig]
        else:
            index_to_grb = {i: v for i, v in enumerate(grb_vars)}

        # ---- Apply fixings (tighten bounds) ----
        n_fixed = 0
        for idx, val in fixings_dict.items():
            if idx in index_to_grb:
                v = index_to_grb[idx]
                v.LB = val
                v.UB = val
                n_fixed += 1

        # ---- MIP Start ----
        if mip_start_dict:
            for idx, val in mip_start_dict.items():
                if idx in index_to_grb:
                    index_to_grb[idx].Start = val

        model.optimize()

        status_map = {
            GRB.OPTIMAL: 'optimal',
            GRB.INFEASIBLE: 'infeasible',
            GRB.INF_OR_UNBD: 'infeasible',
            GRB.UNBOUNDED: 'unbounded',
            GRB.TIME_LIMIT: 'timelimit',
        }
        status = status_map.get(model.Status, str(model.Status))

        info = {
            'solving_time': model.Runtime,
            'n_nodes': int(model.NodeCount) if hasattr(model, 'NodeCount') else 0,
            'n_fixed': n_fixed,
            'n_total_vars': len(grb_vars),
        }

        if model.SolCount > 0:
            solution = {v.VarName: v.X for v in grb_vars}
            obj_val = model.ObjVal
            # MIP gap and constraint info
            try:
                info['mip_gap'] = model.MIPGap
            except Exception:
                info['mip_gap'] = 0.0
            info['feasible'] = True
            # Check constraint violations
            try:
                max_viol = model.MaxVio
                info['max_violation'] = max_viol
            except Exception:
                info['max_violation'] = 0.0
            if status == 'timelimit':
                status = 'timelimit*'  # timelimit but has feasible solution
            return solution, status, obj_val, info
        else:
            info['mip_gap'] = None
            info['feasible'] = False
            info['max_violation'] = None
            return None, status, None, info

    # ------------------------------------------------------------------
    # SCIP solver
    # ------------------------------------------------------------------

    def _solve_scip(self, instance_path, fixings_dict, var_names=None,
                    mip_start_dict=None):
        """
        Solve a MILP instance with SCIP (pyscipopt), applying fixings.

        Parameters and returns same as _solve_gurobi.
        """
        from pyscipopt import Model

        model = Model()
        model.setParam('display/verblevel', 4 if self.verbose else 0)
        model.setParam('limits/time', self.time_limit)
        model.readProblem(instance_path)

        scip_vars = model.getVars()

        if var_names is not None:
            var_by_name = {v.name: v for v in scip_vars}
            index_to_scip = {}
            for idx, name in enumerate(var_names):
                if name in var_by_name:
                    index_to_scip[idx] = var_by_name[name]
        else:
            index_to_scip = {i: v for i, v in enumerate(scip_vars)}

        # ---- Apply fixings ----
        n_fixed = 0
        for idx, val in fixings_dict.items():
            if idx in index_to_scip:
                v = index_to_scip[idx]
                model.fixVar(v, val)
                n_fixed += 1

        # ---- MIP Start (SCIP partial solution) ----
        if mip_start_dict:
            sol = model.createSol()
            for idx, val in mip_start_dict.items():
                if idx in index_to_scip:
                    model.setSolVal(sol, index_to_scip[idx], val)
            accepted = model.trySol(sol)

        model.optimize()

        scip_status = model.getStatus()
        status_map = {
            'optimal': 'optimal',
            'infeasible': 'infeasible',
            'unbounded': 'unbounded',
            'timelimit': 'timelimit',
        }
        status = status_map.get(scip_status, scip_status)

        info = {
            'solving_time': model.getSolvingTime(),
            'n_nodes': model.getNNodes(),
            'n_fixed': n_fixed,
            'n_total_vars': len(scip_vars),
        }

        if model.getNSols() > 0:
            best_sol = model.getBestSol()
            solution = {v.name: model.getSolVal(best_sol, v) for v in scip_vars}
            obj_val = model.getSolObjVal(best_sol)
            if status == 'timelimit':
                status = 'feasible'
            return solution, status, obj_val, info
        else:
            return None, status, None, info

    # ------------------------------------------------------------------
    # Unified solve interface
    # ------------------------------------------------------------------

    def _solve(self, instance_path, fixings_dict, var_names=None,
               mip_start_dict=None):
        """Route to Gurobi or SCIP based on self.solver."""
        if self.solver == 'gurobi':
            return self._solve_gurobi(
                instance_path, fixings_dict, var_names, mip_start_dict)
        elif self.solver == 'scip':
            return self._solve_scip(
                instance_path, fixings_dict, var_names, mip_start_dict)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def solve_with_fixings(self, instance_path, result, n_vars, var_types,
                           var_names=None, use_mip_start=False):
        """
        One-shot solve: extract fixings from predictions and solve.

        Parameters
        ----------
        instance_path : str
            Path to MILP instance file (.lp / .mps).
        result : dict
            Output from DeslicingDecoder.forward().
        n_vars : int
            Total number of variables.
        var_types : LongTensor [N]
            Variable types.
        var_names : list of str, optional
            Variable names (matching solver's variable ordering).
        use_mip_start : bool
            If True, also inject full prediction as MIP Start.

        Returns
        -------
        solution : dict or None
        status : str
        obj_val : float or None
        info : dict
        """
        fixings_ordered = self.extract_fixings(result, n_vars, var_types)
        fixings_dict = {idx: val for idx, (val, _) in fixings_ordered.items()}

        mip_start = None
        if use_mip_start:
            mip_start = self.get_full_prediction(result, n_vars)

        return self._solve(instance_path, fixings_dict, var_names, mip_start)

    def backtracking_solve(self, instance_path, result, n_vars, var_types,
                           var_names=None, use_mip_start=True):
        """
        Solve with automatic backtracking on infeasibility.

        Strategy:
            1. Try fixing all confident variables.
            2. If infeasible, keep only the top 50% most confident.
            3. If still infeasible, keep top 25%, then 12.5%, ...
            4. If all attempts fail, solve without any fixings
               (but still with MIP Start if available).

        Parameters
        ----------
        instance_path : str
            Path to MILP instance file.
        result : dict
            Model predictions.
        n_vars : int
            Total variables.
        var_types : LongTensor [N]
            Variable types.
        var_names : list of str, optional
            Variable names.
        use_mip_start : bool
            Inject full prediction as MIP Start.

        Returns
        -------
        solution : dict or None
        status : str
        obj_val : float or None
        info : dict
            Includes 'n_fixed', 'backtrack_step', 'total_candidates'.
        """
        # Extract all candidate fixings sorted by confidence
        all_fixings = self.extract_fixings(result, n_vars, var_types)
        total_candidates = len(all_fixings)

        mip_start = None
        if use_mip_start:
            mip_start = self.get_full_prediction(result, n_vars)

        fixing_list = list(all_fixings.items())  # [(idx, (val, conf)), ...]

        for step in range(self.max_backtrack_steps):
            # Exponential decay: 100% -> 50% -> 25% -> 12.5% -> ...
            ratio = 1.0 / (2 ** step)
            n_fix = max(1, int(len(fixing_list) * ratio))

            if n_fix == 0:
                break

            # Take top-k most confident fixings
            top_fixings = fixing_list[:n_fix]
            fixings_dict = {idx: val for idx, (val, _) in top_fixings}

            sol, status, obj, info = self._solve(
                instance_path, fixings_dict, var_names, mip_start)

            info['backtrack_step'] = step
            info['total_candidates'] = total_candidates

            if sol is not None:
                # Found feasible solution
                return sol, status, obj, info

            if not self.verbose:
                pass  # silently backtrack
            else:
                print(f"  [backtrack] step {step}: {n_fix} fixings "
                      f"-> {status}, retrying with fewer...")

        # ---- Final fallback: no fixings, only MIP Start ----
        sol, status, obj, info = self._solve(
            instance_path, {}, var_names, mip_start)
        info['backtrack_step'] = self.max_backtrack_steps
        info['total_candidates'] = total_candidates
        return sol, status, obj, info

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self, fixings):
        """
        Print a summary of extracted fixings.

        Parameters
        ----------
        fixings : OrderedDict
            Output from extract_fixings().
        """
        if not fixings:
            print("No fixings extracted (model uncertain about all vars).")
            return

        n_fix = len(fixings)
        confidences = [conf for _, (_, conf) in fixings.items()]
        vals = [val for _, (val, _) in fixings.items()]

        n_zeros = sum(1 for v in vals if v == 0)
        n_ones = sum(1 for v in vals if v == 1)
        n_other = n_fix - n_zeros - n_ones

        print(f"Trust Region Fixings: {n_fix} variables")
        print(f"  Fixed to 0: {n_zeros}")
        print(f"  Fixed to 1: {n_ones}")
        if n_other > 0:
            print(f"  Fixed to other values: {n_other}")
        print(f"  Confidence: min={min(confidences):.4f}, "
              f"max={max(confidences):.4f}, "
              f"mean={sum(confidences)/len(confidences):.4f}")
