import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout,config): 
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid,config=config,t_=None)
        self.gc2 = GraphConvolution(nhid, out,config=config)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class s_GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout,config): 
        super(s_GCN, self).__init__()
        if config.spatial_adp:
            self.gc1 = GraphConvolution(nfeat, nhid,config=config,t_='s')
            self.gc2 = GraphConvolution(nhid, out,config=config)
            self.dropout = dropout
        else:
            self.gc1 = GraphConvolution(nfeat, nhid,config=config)
            self.gc2 = GraphConvolution(nhid, out,config=config)
            self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class f_GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout,config): 
        super(f_GCN, self).__init__()
        if config.feature_adp:
            self.gc1 = GraphConvolution(nfeat, nhid,config=config,t_='f')
            self.gc2 = GraphConvolution(nhid, out,config=config)
            self.dropout = dropout
        else:
            self.gc1 = GraphConvolution(nfeat, nhid,config=config)
            self.gc2 = GraphConvolution(nhid, out,config=config)
            self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class STMGAMF(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout,config):
        super(STMGAMF, self).__init__()
        self.SGCN = s_GCN(nfeat, nhid1, nhid2, dropout,config)
        self.FGCN = f_GCN(nfeat, nhid1, nhid2, dropout,config)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout,config)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )
        self.config = config
        self.c_f = nn.Parameter(torch.tensor(1,dtype=torch.float32),requires_grad=True)
        self.c_s = nn.Parameter(torch.tensor(1,dtype=torch.float32),requires_grad=True)
        self.c_com = nn.Parameter(torch.tensor(1,dtype=torch.float32),requires_grad=True)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        com1 = self.CGCN(x, sadj)  # Co_GCN
        com2 = self.CGCN(x, fadj)  # Co_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN

        emb = torch.stack([emb1, (com1 + com2) / 2, emb2], dim=1)
        emb, att = self.att(emb)
        emb1 = torch.tanh(self.c_f)*emb1 + torch.tanh(self.c_s)*emb2 + torch.tanh(self.c_com)*((com1 + com2)/2)
        if self.config.f_cat:
            emb = self.MLP(emb1+emb)
        else:
            emb = self.MLP(emb1)

        [pi, disp, mean] = self.ZINB(emb)
        return com1, com2, emb, pi, disp, mean
