import os.path as osp
import argparse
import numpy as np 
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.data import Dataset, download_url

from utils2 import load_data_adjacency_n_greatest_clusters

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
# roi = 'BRAC4002.3c_ROI1_MRTX.'

# S, features , labels , edge_index , edge_weight=load_data_adjacency_n_greatest_clusters(roi)

# edge_index = torch.LongTensor(edge_index)

# data_list = [Data(features,edge_index=edge_index,edge_attr=edge_weight,y=labels)]
# data= data_list[0]


# n = labels.shape[0]

# idx_train = range((int(np.floor(0.7*n)))) 
# idx_val = range((int(np.floor(0.7*n))), (int(np.floor(0.8*n))))
# idx_test = range((int(np.floor(0.8*n))), (int(np.floor(1*n))))

# train_mask = torch.tensor([i in idx_train for i in range(n)])
# val_mask = torch.tensor([i in idx_val for i in range(n)])
# test_mask = torch.tensor([i in idx_test for i in range(n)])

# data.train_mask = train_mask
# data.val_mask = val_mask
# data.test_mask = test_mask

# print(data_list)
# data.num_classes= len(labels.unique())
# print(data.num_classes)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 64, cached=True
                             )
        self.conv2 = GCNConv(64, data.num_classes, cached=True
                            )
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x.float(), edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x.float(), edge_index, edge_weight)
        return F.log_softmax(x, dim=1)




def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


rois = ['BRAC3495.3f_ROI1_Cont_crop1.', 'BRAC3495.3f_ROI1_Cont_crop2.',
    'BRAC3326.4e_ROI1_Cont_crop1.', 'BRAC3438.6f_ROI1_Cont.', 'BRAC3438.6f_ROI2_Cont.', 'BRAC3438.6f_ROI3_Cont.', 'BRAC3529.2d_ROI1_MRTX.', 'BRAC4002.3c_ROI2_MRTX_crop1.', 'BRAC4002.3c_ROI2_MRTX_crop2.', 'BRAC4002.3c_ROI3_MRTX.', 'BRAC3529.2b_ROI1_MRTX_crop2.', 'BRAC4002.3c_ROI1_MRTX.'][:6]

for roi in rois:
    print(f"ROI: {roi}")
    S, features , labels , edge_index , edge_weight=load_data_adjacency_n_greatest_clusters(roi,n_clusters=15)
    edge_index = torch.LongTensor(edge_index)

    data_list = [Data(features,edge_index=edge_index,edge_attr=edge_weight,y=labels)]
    data= data_list[0]
    n = labels.shape[0]

    idx_train = range((int(np.floor(0.7*n)))) 
    idx_val = range((int(np.floor(0.7*n))), (int(np.floor(0.8*n))))
    idx_test = range((int(np.floor(0.8*n))), (int(np.floor(1*n))))

    train_mask = torch.tensor([i in idx_train for i in range(n)])
    val_mask = torch.tensor([i in idx_val for i in range(n)])
    test_mask = torch.tensor([i in idx_test for i in range(n)])

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # print(data_list)
    data.num_classes= len(labels.unique())
    # print(data.num_classes)

    values = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    # print(data.edge_index)

    # print(data.edge_index.shape)
    # print(type(data.edge_index[0][0]))


    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.


    best_val_acc = test_acc = 0
    for epoch in range(1, 100):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        values.append(train_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))



    plt.plot(values)
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.title(f"ROI: {roi}")
    plt.savefig(f'stolen/15 largest clusters - ROI: {roi}, (ELU,Sigmoid,ELU,SM), Test Accuracy: {test_acc}.png')
    plt.clf()