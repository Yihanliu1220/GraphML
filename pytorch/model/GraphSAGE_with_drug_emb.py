import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class DrugEmbeddingProjector(nn.Module):
    """
    Project high-dimensional external drug embeddings to KnowDDI's emb_dim.
    """

    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=32, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)


class GraphSAGEWithDrugEmb(nn.Module):
    """
    GraphSAGE variant with mixed node features:
    - nodes with external drug embeddings use projected fixed features
    - all other nodes use learnable embeddings
    """

    def __init__(self, params):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(params.gcn_dropout)
        self.activation = F.relu
        self.aggregator_type = params.gcn_aggregator_type
        self.num_nodes = params.num_nodes
        self.num_rels = params.aug_num_rels
        self.emb_dim = params.emb_dim
        self.num_gcn_layers = params.num_gcn_layers

        self.num_drug_nodes = getattr(params, 'num_drug_nodes', 1710)
        self.use_drug_embeddings = getattr(params, 'use_drug_embeddings', True)

        self.trainable_embed = nn.Parameter(
            torch.Tensor(self.num_nodes, self.emb_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.trainable_embed, gain=nn.init.calculate_gain('relu'))

        if self.use_drug_embeddings:
            drug_emb_input_dim = getattr(params, 'drug_emb_input_dim', 4096)
            drug_emb_hidden_dim = getattr(params, 'drug_emb_hidden_dim', 512)
            drug_emb_dropout = getattr(params, 'drug_emb_dropout', 0.1)
            self.drug_projector = DrugEmbeddingProjector(
                input_dim=drug_emb_input_dim,
                hidden_dim=drug_emb_hidden_dim,
                output_dim=self.emb_dim,
                dropout=drug_emb_dropout,
            )
            self.register_buffer('fixed_drug_embeddings', None)
            self.register_buffer('has_embedding_mask', None)
            self.register_buffer('embedding_lookup', None)
        else:
            self.drug_projector = None
            self.register_buffer('fixed_drug_embeddings', None)
            self.register_buffer('has_embedding_mask', None)
            self.register_buffer('embedding_lookup', None)

        self.rel_weght = nn.Parameter(torch.Tensor(self.num_rels, self.emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.rel_weght, gain=nn.init.calculate_gain('relu'))

        for _ in range(self.num_gcn_layers):
            self.layers.append(SAGEConv(self.emb_dim, self.emb_dim, self.aggregator_type))

    def load_drug_embeddings(self, embeddings_path):
        """
        Accept either:
        - a plain tensor of shape (num_drugs, dim)
        - a dict with key 'drug_embeddings' and optional metadata:
          'ordered_node_indices', 'has_embedding_mask', 'num_drug_nodes', 'embedding_dim'
        """
        loaded = torch.load(embeddings_path, map_location='cpu')
        if isinstance(loaded, dict):
            if 'drug_embeddings' not in loaded:
                raise KeyError("Expected key 'drug_embeddings' in embedding file.")
            drug_embeddings = loaded['drug_embeddings']
            ordered_node_indices = loaded.get('ordered_node_indices')
            has_embedding_mask = loaded.get('has_embedding_mask')
            num_drug_nodes = loaded.get('num_drug_nodes')
        elif torch.is_tensor(loaded):
            drug_embeddings = loaded
            ordered_node_indices = None
            has_embedding_mask = None
            num_drug_nodes = None
        else:
            raise TypeError('Embedding file must contain a tensor or a dict.')

        drug_embeddings = drug_embeddings.float()
        if drug_embeddings.dim() != 2:
            raise ValueError('Drug embedding matrix must be 2D: (num_drugs, embedding_dim).')

        self.fixed_drug_embeddings = drug_embeddings
        num_loaded, input_dim = drug_embeddings.shape

        if ordered_node_indices is not None:
            ordered_node_indices = torch.as_tensor(ordered_node_indices, dtype=torch.long)
            if ordered_node_indices.numel() != num_loaded:
                raise ValueError('ordered_node_indices length must match number of embeddings.')
            node_indices = ordered_node_indices
        else:
            if has_embedding_mask is not None:
                mask_prefix = torch.as_tensor(has_embedding_mask, dtype=torch.bool)
                node_indices = torch.nonzero(mask_prefix, as_tuple=False).squeeze(1)
                if node_indices.numel() != num_loaded:
                    raise ValueError(
                        'Number of True values in has_embedding_mask must match number of embeddings.'
                    )
            else:
                inferred = num_drug_nodes if num_drug_nodes is not None else num_loaded
                inferred = min(inferred, num_loaded, self.num_nodes)
                node_indices = torch.arange(inferred, dtype=torch.long)

        if node_indices.numel() > self.num_nodes:
            raise ValueError('Embedding node indices exceed total graph node count.')

        full_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        full_mask[node_indices] = True

        lookup = torch.full((self.num_nodes,), -1, dtype=torch.long)
        lookup[node_indices] = torch.arange(node_indices.numel(), dtype=torch.long)

        self.has_embedding_mask = full_mask
        self.embedding_lookup = lookup
        self.num_drug_nodes = int(node_indices.numel())

        if input_dim != self.drug_projector.projector[0].in_features:
            raise ValueError(
                f'Embedding dim mismatch: file has {input_dim}, '
                f'but model expects {self.drug_projector.projector[0].in_features}.'
            )

    def get_node_embeddings(self, node_indices):
        if not self.use_drug_embeddings or self.has_embedding_mask is None:
            return self.trainable_embed[node_indices]

        embeddings = self.trainable_embed[node_indices].clone()
        has_embedding = self.has_embedding_mask[node_indices]

        if has_embedding.any():
            lookup_indices = self.embedding_lookup[node_indices[has_embedding]]
            fixed_emb = self.fixed_drug_embeddings[lookup_indices]
            projected_emb = self.drug_projector(fixed_emb)
            embeddings[has_embedding] = projected_emb

        return embeddings

    def forward(self, g):
        h = self.get_node_embeddings(g.ndata['idx'])
        h = self.dropout(h)
        edge_weight = self.rel_weght[g.edata['type']]

        for l, layer in enumerate(self.layers):
            g.ndata['h'] = layer(g, h, edge_weight=edge_weight)
            if l != len(self.layers) - 1:
                g.ndata['h'] = self.activation(g.ndata['h'])
                g.ndata['h'] = self.dropout(g.ndata['h'])
            h = g.ndata['h']

            if l == 0:
                initial_emb = self.get_node_embeddings(g.ndata['idx'])
                x = torch.cat([initial_emb, g.ndata['h']], dim=1)
                g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.emb_dim)
            else:
                g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

        return g.ndata.pop('h')


GraphSAGE = GraphSAGEWithDrugEmb
