import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau


from torch_geometric.nn import GCNConv, GINEConv, GraphConv

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import  softmax

torch.set_printoptions(profile="full")


class GlobalAttentionPool(torch.nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


class GIN_AC(torch.nn.Module):
    def __init__(self, node_dim = 52, edge_dim = 10, dropout=0, conv_dim=256, graph_dim=256, num_layer = 4, protein_outdim = 512, 
                        gnn_operator = 'gin', pool = 'attention', protein_dim = 1024, prot_layer = 2):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GIN_AC, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.conv_dim = conv_dim
        self.dropout = dropout
        self.gnn_operator = gnn_operator
        self.prot_layer = prot_layer
        self.pool_type = pool
        self.graph_dim = graph_dim
        self.prot_dim = protein_dim
        self.prot_outdim = protein_outdim
        self.num_gc_layers = num_layer
        self.setup_layers()


    def setup_layers(self):
        """
        Creating the layers.
        """

        if self.gnn_operator == "gcn":
            self.convolution_1 = GCNConv(self.node_dim, self.conv_dim)
            self.convolution_2 = GCNConv(self.conv_dim, self.conv_dim)
            self.convolution_3 = GCNConv(self.conv_dim, self.conv_dim)
        elif self.gnn_operator == "gin":
            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            for i in range(self.num_gc_layers):
                
                if i == 0:
                    nn = torch.nn.Sequential(torch.nn.Linear(self.node_dim, self.conv_dim), torch.nn.ReLU(), torch.nn.Linear(self.conv_dim, self.conv_dim))
                else:
                    nn = torch.nn.Sequential(torch.nn.Linear(self.conv_dim, self.conv_dim), torch.nn.ReLU(), torch.nn.Linear(self.conv_dim, self.conv_dim))
                conv = GINEConv(nn, train_eps=True, edge_dim=self.edge_dim)
                bn = torch.nn.BatchNorm1d(self.conv_dim)

                self.convs.append(conv)
                self.bns.append(bn)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        if self.pool_type == 'mean':
            self.pool = global_mean_pool
        elif self.pool_type == 'add':
            self.pool = global_add_pool
        elif self.pool_type == 'max':
            self.pool = global_max_pool
        elif self.pool_type == 'attention':
            self.pool = GlobalAttentionPool(self.conv_dim)
        
        else:
            raise ValueError('Not defined pooling!')

        self.trans_prot = torch.nn.ModuleList()
        for i in range(self.prot_layer):
            if i == 0:
                self.trans_prot.append(torch.nn.Linear(self.prot_dim, self.prot_outdim))
            else:
                self.trans_prot.append(torch.nn.Linear(self.prot_outdim, self.prot_outdim))
            self.trans_prot.append(torch.nn.ReLU())

        self.out_lin = torch.nn.Sequential(
            torch.nn.Linear(self.conv_dim * self.num_gc_layers, self.graph_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.graph_dim, self.graph_dim)
            )

    def forward(self, data, prot_feat, with_prot = False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """

        node_features = data.x
        edge_index_1 = data.edge_index.to(dtype=torch.long)
        prot_feat = prot_feat.view(1, -1)

        batch_1 = (
            data.batch
            if hasattr(data, "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data.num_nodes)
        )

        xs = []
        for i in range(self.num_gc_layers):
            
            # old
            node_features = F.relu(self.convs[i](node_features, edge_index_1, data.edge_attr))
            node_features = self.bns[i](node_features)
            xs.append(node_features)

        graph_feature_1 = [self.pool(x, data.batch) for x in xs]

        graph_feature_2 = torch.cat(graph_feature_1, 1)
        latent_feature = self.out_lin(graph_feature_2)

        if with_prot:
            for i in range(self.prot_layer):
                prot_feat = self.trans_prot[i](prot_feat)
            return latent_feature, prot_feat
        else:
            return latent_feature

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class Prot_trans(torch.nn.Module):
    def __init__(self, prot_layer = 2, prot_dim=1280, latent_dim=256):
        super(Prot_trans, self).__init__()

        self.prot_layer = prot_layer
        self.prot_dim = prot_dim
        self.latent_dim = latent_dim
        self.trans_prot = torch.nn.ModuleList()
        for i in range(self.prot_layer):
            if i == 0:
                self.trans_prot.append(torch.nn.Linear(self.prot_dim, self.latent_dim))
            else:
                self.trans_prot.append(torch.nn.Linear(self.latent_dim, self.latent_dim))
            self.trans_prot.append(torch.nn.ReLU())
    def forward(self, prot_feat):
        prot_feat = prot_feat.view(1, -1)
        for i in range(self.prot_layer):
            prot_feat = self.trans_prot[i](prot_feat)
        return prot_feat


