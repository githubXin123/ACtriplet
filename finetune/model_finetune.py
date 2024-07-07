import random
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINEConv, GraphConv

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import  softmax
from torch_geometric.utils import to_dense_batch


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
    def __init__(self, node_dim = 52, edge_dim = 10, dropout=0, conv_dim=256, graph_dim=512, protein_dim=1280, num_layer = 4,
                        gnn_operator = 'gin', pool = 'add', pred_act = 'Elu', prot_layer = 2, DMA_depth = 2, pred_n_layer=2):
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
        self.pool_type = pool
        self.graph_dim = graph_dim
        self.protein_dim = protein_dim
        self.num_gc_layers = num_layer
        self.prot_layer = prot_layer
        self.pred_init_dim = self.graph_dim * self.num_gc_layers + self.graph_dim
        self.DMA_depth = DMA_depth
        self.pred_n_layer = pred_n_layer
        self.pred_act = pred_act
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
        self.out_lin = torch.nn.Sequential(
            torch.nn.Linear(self.conv_dim * self.num_gc_layers, self.graph_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.graph_dim, self.graph_dim)
            )
        self.super_final = torch.nn.Linear(self.graph_dim, self.graph_dim)

        
        if self.pred_act == 'PRelu':
            act_layer = torch.nn.PReLU()
        elif self.pred_act == 'Relu':
            act_layer = torch.nn.ReLU(inplace=True)
        elif self.pred_act == 'Elu':
            act_layer = torch.nn.ELU(inplace=True)
        elif self.pred_act == 'LeakRelu':
            act_layer = torch.nn.LeakyReLU(inplace=True)
        elif self.pred_act == 'Softplus':
            act_layer = torch.nn.Softplus()
        else:
            print('pred active not found')

        self.trans_prot_ft = torch.nn.ModuleList()
        for i in range(self.prot_layer):
            if i == 0:
                self.trans_prot_ft.append(torch.nn.Linear(self.protein_dim, self.graph_dim))
            else:
                self.trans_prot_ft.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.trans_prot_ft.append(torch.nn.ReLU())

        self.c_to_p_transform = torch.nn.ModuleList()
        self.p_to_c_transform = torch.nn.ModuleList()
        self.mc1 = torch.nn.ModuleList()
        self.mp1 = torch.nn.ModuleList()
        self.hc0 = torch.nn.ModuleList()
        self.hp0 = torch.nn.ModuleList()
        self.hc1 = torch.nn.ModuleList()
        self.hp1 = torch.nn.ModuleList()

        for i in range(self.DMA_depth):
            self.c_to_p_transform.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.p_to_c_transform.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.mc1.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.mp1.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.hc0.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.hp0.append(torch.nn.Linear(self.graph_dim, self.graph_dim))
            self.hc1.append(torch.nn.Linear(self.graph_dim, 1))
            self.hp1.append(torch.nn.Linear(self.graph_dim, 1))


        self.GRU_dma = torch.nn.GRUCell(self.graph_dim, self.graph_dim)

        pred_head = [
                torch.nn.Linear(self.graph_dim * self.graph_dim * 2, self.graph_dim), 
                act_layer,
            ]
        for _ in range(self.pred_n_layer):
            pred_head.extend([
                torch.nn.Linear(self.graph_dim, self.graph_dim), 
                act_layer,
            ])
        pred_head.append(torch.nn.Linear(self.graph_dim, 1))
        self.W_out = torch.nn.Sequential(*pred_head)

    def mask_softmax(self,a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax

    def Pairwise_pred_module(self, batch_size, comp_feature, prot_feature, vertex_mask, seq_mask):

        pairwise_c_feature = F.elu(comp_feature)  # [batch, max_node, node_dim]
        pairwise_p_feature = F.elu(prot_feature)  # [batch, max_seq, node_dim]

        pairwise_pred = torch.sigmoid(torch.matmul(pairwise_c_feature, pairwise_p_feature.transpose(1,2)))
        pairwise_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))
        pairwise_pred = pairwise_pred * pairwise_mask
		
        return pairwise_pred    # [batch, max_node, max_seq] 

    def Affinity_pred_module(self, batch_size, comp_feature, prot_feature, super_feature, vertex_mask, seq_mask, pairwise_pred):
		
        comp_feature = F.elu(comp_feature)
        prot_feature = F.elu(prot_feature)

        # [batch, dim]
        super_feature = F.elu(self.super_final(super_feature.view(batch_size,-1)))

        cf, pf = self.dma_gru(batch_size, comp_feature, vertex_mask, prot_feature, seq_mask, pairwise_pred)

        # [batch, 512]
        cf = torch.cat([cf.view(batch_size,-1), super_feature.view(batch_size,-1)], dim=1)
        # [batch, 512 * 256]
        kroneck = F.elu(torch.matmul(cf.view(batch_size,-1,1), pf.view(batch_size,1,-1)).view(batch_size,-1))

        affinity_pred = self.W_out(kroneck)
        return affinity_pred

    def dma_gru(self, batch_size, comp_feats, vertex_mask, prot_feats, seq_mask, pairwise_pred):
        vertex_mask = vertex_mask.view(batch_size,-1,1)     # [batch_size, max_node, 1]
        seq_mask = seq_mask.view(batch_size,-1,1)           # [batch_size, max_seq, 1]

        c0 = torch.sum(comp_feats*vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        p0 = torch.sum(prot_feats*seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        m = c0*p0   #[batch_size, dim]
        for DMA_iter in range(self.DMA_depth):
            c_to_p = torch.matmul(pairwise_pred.transpose(1,2), F.tanh(self.c_to_p_transform[DMA_iter](comp_feats)))
            p_to_c = torch.matmul(pairwise_pred, F.tanh(self.p_to_c_transform[DMA_iter](prot_feats)))

            c_tmp = F.tanh(self.hc0[DMA_iter](comp_feats))*F.tanh(self.mc1[DMA_iter](m)).view(batch_size,1,-1)*p_to_c
            p_tmp = F.tanh(self.hp0[DMA_iter](prot_feats))*F.tanh(self.mp1[DMA_iter](m)).view(batch_size,1,-1)*c_to_p
            c_att = self.mask_softmax(self.hc1[DMA_iter](c_tmp).view(batch_size,-1), vertex_mask.view(batch_size,-1)) 
            p_att = self.mask_softmax(self.hp1[DMA_iter](p_tmp).view(batch_size,-1), seq_mask.view(batch_size,-1))
            cf = torch.sum(comp_feats*c_att.view(batch_size,-1,1), dim=1)
            pf = torch.sum(prot_feats*p_att.view(batch_size,-1,1), dim=1)
            
            m = self.GRU_dma(m, cf*pf)

        return cf, pf

    def forward(self, data, prot_feat):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """

        node_features = data.x
        edge_index_1 = data.edge_index.to(dtype=torch.long)
        prot_feat = prot_feat.to(node_features.device)
        xs = []
        for i in range(self.num_gc_layers):

            node_features = F.relu(self.convs[i](node_features, edge_index_1, data.edge_attr))
            node_features = self.bns[i](node_features)
            xs.append(node_features)
        graph_feature_1 = [self.pool(x, data.batch) for x in xs]

        graph_feature_2 = torch.cat(graph_feature_1, 1) # [batch_sieze, 1024]
        graph_feature_2 = self.out_lin(graph_feature_2) # [batch_sieze, 1024]
        graph_feature_2 = graph_feature_2.unsqueeze(1)  # [batch_sieze, 1, 1024]
        batch_size = graph_feature_2.size(0)

        
        for i in range(self.prot_layer):
            prot_feat = self.trans_prot_ft[i](prot_feat)

        prot_feat = prot_feat.unsqueeze(0)
        prot_feat  = prot_feat.expand(graph_feature_2.size(0), -1, -1)
        prot_mask = torch.ones(batch_size, prot_feat.size(1)).to(node_features.device)
        atom_feat_padding, atom_mask = to_dense_batch(node_features, data.batch)
        atom_mask = atom_mask.type(torch.float)

        pairwise_pred = self.Pairwise_pred_module(batch_size, atom_feat_padding, prot_feat, atom_mask, prot_mask)

        affinity_pred = self.Affinity_pred_module(batch_size, atom_feat_padding, prot_feat, graph_feature_2, atom_mask, prot_mask, pairwise_pred)

        return affinity_pred, pairwise_pred

    def load_my_state_dict(self, state_dict, display):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if display:
                print('the layer being loaded:', name)
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
