import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from  torch_geometric.nn import GCNConv, Sequential


class GCN(nn.Module):
    def __init__(self, num_feat, num_hid, num_class,dropout):

        super(GCN, self).__init__()
        self.gc1 = GCNConv(num_feat, num_hid)
        self.gc2 = GCNConv(num_hid, num_hid)
        self.gc3 = GCNConv(num_hid, num_hid)
        self.gc4 = GCNConv(num_hid, num_class)
        # self.gc5 = GCNConv(num_hid, num_class)
        self.dropout = dropout

    def forward(self,x,edge_index,edge_weight): 
        x= F.elu(self.gc1(x=x, edge_index= edge_index , edge_weight = edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.gc2(x=x.float(),edge_index= edge_index, edge_weight= edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc3(x=x.float(),edge_index= edge_index, edge_weight= edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x=x.float(),edge_index= edge_index, edge_weight= edge_weight)
        return F.log_softmax(x, dim=1)
    
    