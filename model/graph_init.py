import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from model.model import PreNormLayer, PreNormException, BaseModel


class BipartiteGCNConv(torch_geometric.nn.MessagePassing):
    """
    A single bipartite graph convolution layer.

    Message passing: for each target node j, aggregate messages from all
    source nodes i connected via edges (i, j):
        m_{i→j} = MLP( W_src · h_i + W_edge · e_{ij} + W_dst · h_j )
    Then update:
        h_j' = MLP( [agg(m_{i→j}); h_j] )

    This follows the same pattern as the original BipartiteGraphConvolution
    but with configurable embedding size.
    """

    def __init__(self, emb_size=64):
        super().__init__('add')

        self.feature_module_left = nn.Sequential(
            nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = nn.Sequential(
            nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = nn.Sequential(
            nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        Parameters
        ----------
        left_features : [N_left, D]  (source nodes)
        edge_indices : [2, E]  (edge_indices[0]=left idx, edge_indices[1]=right idx)
        edge_features : [E, 1]
        right_features : [N_right, D]  (target nodes, to be updated)

        Returns
        -------
        updated right_features : [N_right, D]
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class GraphInitialization(BaseModel):
    """
    Graph Feature Fusion & State Initialization.

    Uses a lightweight bipartite GNN to inject constraint information into
    variable node embeddings. This serves as the "preparation step" before
    entering a latent space — the initial mathematical structure of the MILP
    (constraint matrix A) is fully captured by the bipartite graph.

    Architecture:
        1. Embed raw features (constraints, edges, variables) into D-dim space
        2. Perform K rounds of bidirectional bipartite GCN:
           - var → con: constraints absorb variable information
           - con → var: variables absorb constraint information
           Each round uses residual connections + LayerNorm for stable training.
        3. Output z_var_0: [N_var, D], where each variable embedding encodes
           both its own features and the "pull" from all connected constraints.

    Training tricks included:
        - PreNormLayer: data-driven input normalization (from original codebase)
        - Residual connections: prevent degradation in deeper GNNs
        - LayerNorm: stabilize training across varying graph sizes
        - Dropout: regularization to prevent overfitting
        - Xavier initialization: better gradient flow at init
    """

    def __init__(self, cons_nfeats=5, edge_nfeats=1, var_nfeats=23,
                 emb_size=64, n_conv_rounds=2, dropout=0.1):
        """
        Parameters
        ----------
        cons_nfeats : int
            Number of raw constraint features (default: 5 from ecole).
        edge_nfeats : int
            Number of raw edge features (default: 1, the coefficient value).
        var_nfeats : int
            Number of raw variable features (default: 23, ecole 19 + 4 bounds).
        emb_size : int
            Hidden dimension D for all embeddings and GCN layers.
        n_conv_rounds : int
            Number of bipartite GCN rounds (bidirectional message passing).
        dropout : float
            Dropout rate applied after each GCN round.
        """
        super().__init__()
        self.emb_size = emb_size
        self.n_conv_rounds = n_conv_rounds

        # ── Feature Embedding ──
        self.cons_embedding = nn.Sequential(
            PreNormLayer(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        self.var_embedding = nn.Sequential(
            PreNormLayer(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )

        # ── Bipartite GCN Layers (K rounds, each with var→con and con→var) ──
        self.conv_v_to_c = nn.ModuleList([
            BipartiteGCNConv(emb_size) for _ in range(n_conv_rounds)
        ])
        self.conv_c_to_v = nn.ModuleList([
            BipartiteGCNConv(emb_size) for _ in range(n_conv_rounds)
        ])

        # ── LayerNorm for residual connections ──
        self.layernorm_var = nn.ModuleList([
            nn.LayerNorm(emb_size) for _ in range(n_conv_rounds)
        ])
        self.layernorm_con = nn.ModuleList([
            nn.LayerNorm(emb_size) for _ in range(n_conv_rounds)
        ])

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout)

        # ── Initialize weights ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features):
        """
        Parameters
        ----------
        constraint_features : Tensor [N_con, cons_nfeats]
            Raw constraint node features.
        edge_indices : LongTensor [2, E]
            Bipartite edge indices. Row 0 = constraint index, row 1 = variable index.
        edge_features : Tensor [E, 1]
            Edge features (constraint matrix coefficients).
        variable_features : Tensor [N_var, var_nfeats]
            Raw variable node features.

        Returns
        -------
        z_var_0 : Tensor [N_var, emb_size]
            Initialized variable embeddings enriched with constraint information.
        """
        # Edge indices: con→var (original) and var→con (reversed)
        # edge_indices[0] = constraint idx, edge_indices[1] = variable idx
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0
        )

        # ── Step 1: Embed raw features into D-dim space ──
        cons = self.cons_embedding(constraint_features)
        edge = self.edge_embedding(edge_features)
        var = self.var_embedding(variable_features)

        # ── Step 2: K rounds of bipartite message passing ──
        for k in range(self.n_conv_rounds):
            # var → con: constraints absorb variable info
            # source=var, target=con, edges go from var to con
            cons_new = self.conv_v_to_c[k](
                var, reversed_edge_indices, edge, cons
            )
            cons = self.layernorm_con[k](cons + self.dropout(cons_new))

            # con → var: variables absorb constraint info
            # source=con, target=var, edges go from con to var
            var_new = self.conv_c_to_v[k](
                cons, edge_indices, edge, var
            )
            var = self.layernorm_var[k](var + self.dropout(var_new))

        # ── Step 3: z_var_0 ──
        return var
