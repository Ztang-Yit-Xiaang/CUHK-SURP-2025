import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
 
 
class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
 
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
 
    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in
 
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
 
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
 
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
 
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPWithAttention, self).__init__()
        self.fc_mt = nn.Linear(1, hidden_dim)
        self.fc_E = nn.Linear(3, hidden_dim)
        self.fc_L = nn.Linear(3, hidden_dim)
        self.fc_cs = nn.Linear(1, hidden_dim)
        self.attention = MultiHeadSelfAttention(hidden_dim, hidden_dim, hidden_dim, 4)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(hidden_dim*4, hidden_dim)
        
        # self.fc_out = nn.Linear(4, 1)
        self.cs_range = [0.5,2]
        self.E_range = [5,50]
        self.L_range = [0,30]
        self.mT_range = [0,120]
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def normalize(self, data, data_range):
        return (data-data_range[0])/(data_range[1]-data_range[0])

    def forward(self, x):
        mt = x['mt'].unsqueeze(1).float()
        E = torch.stack(x['E'], dim=1).float()
        L = torch.stack(x['L'], dim=1).float()
        cs = x['cs'].unsqueeze(1).float()
        
        mt = self.normalize(mt, self.mT_range)
        E = self.normalize(E, self.E_range)
        L = self.normalize(L, self.L_range)
        cs = self.normalize(cs, self.cs_range)
        
        # print(mt.shape, E.shape, L.shape, cs.shape)
        
        mt_emb = F.sigmoid(self.fc_mt(mt))
        E_emb = F.sigmoid(self.fc_E(E))
        L_emb = F.sigmoid(self.fc_L(L))
        cs_emb = F.sigmoid(self.fc_cs(cs))
        # emb_seq = torch.stack([mt_emb, E_emb, L_emb, cs_emb], dim=1)
        emb_seq = torch.concatenate([mt_emb, E_emb, L_emb, cs_emb], dim=1)
        
        # att_out = self.attention(emb_seq)
        out = F.sigmoid(self.fc(emb_seq))
        # out = F.sigmoid(self.fc(att_out))

        # out = F.sigmoid(self.fc_out(out.permute(0,2,1))).squeeze(2)
        out = F.sigmoid(self.fc_out(out))
        return out