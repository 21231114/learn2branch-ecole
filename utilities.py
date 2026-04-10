import gzip
import pickle
import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
import argparse

def valid_seed(seed):
    """
    检查种子是否为非负整数。
    """
    seed = int(seed)
    if seed < 0:
        raise argparse.ArgumentTypeError("Seed must be a non-negative integer.")
    return seed
def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class SolutionBipartiteNodeData(torch_geometric.data.Data):
    """
    Data object for the new dataset format (02_generate_dataset.py).
    Stores bipartite graph + solution information (no per-step branching decisions).
    """
    def __init__(self, constraint_features, edge_indices, edge_features,
                 variable_features, sol_values, obj_val):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.sol_values = sol_values    # [N_var] optimal variable values
        self.obj_val = obj_val          # scalar objective value

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor(
                [[self.constraint_features.size(0)],
                 [self.variable_features.size(0)]]
            )
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SolutionGraphDataset(torch_geometric.data.Dataset):
    """
    Dataset for the new data format produced by 02_generate_dataset.py.

    Each sample contains:
        - observation: (constraint_features, (edge_indices, edge_values), variable_features)
        - solution: {status, obj_val, sol_vals, primal_bound, dual_bound, ...}

    Only loads samples that have a feasible solution (sol_vals is not None).
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        constraint_features, (edge_indices, edge_values), variable_features = \
            sample['observation']

        constraint_features = torch.FloatTensor(
            np.asarray(constraint_features, dtype=np.float32)
        )
        edge_indices = torch.LongTensor(
            np.asarray(edge_indices, dtype=np.int64)
        )
        edge_features = torch.FloatTensor(
            np.asarray(edge_values, dtype=np.float32)
        ).unsqueeze(-1)  # [E] -> [E, 1]
        variable_features = torch.FloatTensor(
            np.asarray(variable_features, dtype=np.float32)
        )

        n_vars = variable_features.shape[0]
        solution = sample['solution']

        if solution['sol_vals'] is not None:
            sol_dict = solution['sol_vals']
            sol_values = torch.zeros(n_vars, dtype=torch.float32)
            for i, (name, val) in enumerate(sol_dict.items()):
                if i < n_vars:
                    sol_values[i] = float(val)
            obj_val = torch.tensor(
                float(solution['obj_val']), dtype=torch.float32
            )
        else:
            sol_values = torch.zeros(n_vars, dtype=torch.float32)
            obj_val = torch.tensor(0.0, dtype=torch.float32)

        graph = SolutionBipartiteNodeData(
            constraint_features, edge_indices, edge_features,
            variable_features, sol_values, obj_val,
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


class MultiSolutionGraphDataset(torch_geometric.data.Dataset):
    """
    Dataset that loads K feasible solutions per instance and computes
    Boltzmann-weighted soft targets for training.

    From the Boltzmann energy distribution perspective:
        - Each solution k has objective (energy) f_k
        - Boltzmann weight: w_k = softmax(-f̃_k / T)
          where f̃_k = (f_k - μ) / σ is z-score normalized
        - Soft target: x̄_j = Σ_k w_k · x_{k,j}

    This is equivalent to minimizing KL(π_Boltzmann ∥ p_θ) —
    the model learns a distribution biased toward better solutions
    while benefiting from the diversity of all K solutions.

    Parameters
    ----------
    sample_files : list[str]
        Paths to .pkl sample files (gzipped).
    temperature : float
        Boltzmann temperature T.
        T → 0: only best solution matters (hard label).
        T → ∞: uniform average of all solutions.
        Default 1.0 gives moderate weighting.
    """
    def __init__(self, sample_files, temperature=1.0):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.temperature = temperature

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        constraint_features, (edge_indices, edge_values), variable_features = \
            sample['observation']

        constraint_features = torch.FloatTensor(
            np.asarray(constraint_features, dtype=np.float32))
        edge_indices = torch.LongTensor(
            np.asarray(edge_indices, dtype=np.int64))
        edge_features = torch.FloatTensor(
            np.asarray(edge_values, dtype=np.float32)
        ).unsqueeze(-1)
        variable_features = torch.FloatTensor(
            np.asarray(variable_features, dtype=np.float32))

        n_vars = variable_features.shape[0]
        solution = sample['solution']

        multi_sols = solution.get('multi_sols')   # np.ndarray (n_sols, n_vars)
        multi_objs = solution.get('multi_objs')   # np.ndarray or list (n_sols,)

        if multi_sols is not None and len(multi_sols) > 0:
            multi_sols_t = torch.FloatTensor(
                np.asarray(multi_sols, dtype=np.float32))   # (K, n_vars)
            multi_objs_t = torch.FloatTensor(
                np.asarray(multi_objs, dtype=np.float32))   # (K,)

            # --- Boltzmann weights ---
            # z-score normalize objectives for scale-invariance
            mu = multi_objs_t.mean()
            sigma = multi_objs_t.std()
            if sigma > 1e-8:
                energy = (multi_objs_t - mu) / sigma
            else:
                energy = torch.zeros_like(multi_objs_t)
            boltzmann_w = F.softmax(-energy / self.temperature, dim=0)  # (K,)

            # --- Boltzmann soft target ---
            sol_values = (boltzmann_w.unsqueeze(1) * multi_sols_t).sum(0)  # (n_vars,)

            # best solution (for accuracy metrics)
            best_sol_values = multi_sols_t[0].clone()
            obj_val = torch.tensor(float(multi_objs_t[0]), dtype=torch.float32)

        elif solution.get('sol_vals') is not None:
            # Fallback: single solution (old format)
            sol_dict = solution['sol_vals']
            sol_values = torch.zeros(n_vars, dtype=torch.float32)
            for i, (name, val) in enumerate(sol_dict.items()):
                if i < n_vars:
                    sol_values[i] = float(val)
            best_sol_values = sol_values.clone()
            obj_val = torch.tensor(
                float(solution['obj_val']), dtype=torch.float32)
        else:
            sol_values = torch.zeros(n_vars, dtype=torch.float32)
            best_sol_values = sol_values.clone()
            obj_val = torch.tensor(0.0, dtype=torch.float32)

        graph = SolutionBipartiteNodeData(
            constraint_features, edge_indices, edge_features,
            variable_features, sol_values, obj_val,
        )
        graph.best_sol_values = best_sol_values
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
