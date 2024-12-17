import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv,GATConv

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True,config=None,t_=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adp = config.spatial_adp or config.feature_adp
        if self.adp:
            if t_=='s':
                N = config.size_mat
                indices = torch.tensor(config.sadj,dtype=torch.long)
                values = torch.tensor(torch.ones(config.num_s_edges),requires_grad=True)
                shape = (N, N)
                self.adp_w =  Parameter(torch.sparse_coo_tensor(indices, values, torch.Size(shape))).to(config.device)
            if t_=='f':
                N = config.size_mat
                indices = torch.tensor(config.fadj,dtype=torch.long)
                values = torch.tensor(torch.ones(config.num_f_edges),requires_grad=True)
                shape = (N, N)
                self.adp_w =  Parameter(torch.sparse_coo_tensor(indices, values, torch.Size(shape))).to(config.device)
            if t_ is None:
                self.adp = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.adp:
            output = torch.sparse.mm(self.adp_w,output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

