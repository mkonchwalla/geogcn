import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, load_data_adjacency, load_data_laplacian



from layer import GCN

dropout=0.2
epochs=200
hidden_dim=64
lr = 0.02
weight_decay=0.005
t_factor=1


torch.manual_seed(0)

adj, features, labels , edge_index , edge_weight = load_data_adjacency('BRAC4002.3c_ROI2_MRTX_crop1.')
# S, features, labels , edge_index , edge_weight = load_data_laplacian(t_factor=1)
# S, features, labels , edge_index , edge_weight = load_data()

n = labels.shape[0]

idx_train = range((int(np.floor(0.7*n)))) 
idx_val = range((int(np.floor(0.7*n))), (int(np.floor(0.8*n))))
idx_test = range((int(np.floor(0.8*n))), (int(np.floor(1*n))))

# print(idx_train,idx_val,idx_test)


model = GCN(features.shape[1],
            hidden_dim,
            labels.max().item() + 1,
            dropout)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# optimizer = torch.optim.Adam([
#     dict(params=model.gc1.parameters(), weight_decay=weight_decay),
#     dict(params=model.gc2.parameters(), weight_decay=0),
#     dict(params=model.gc3.parameters(), weight_decay=0)
# ], lr=0.01)

print("Training the model")

for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()

#     print('features/adj',features,adj)

    output = model(features.float(), edge_index, edge_weight)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])

    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

"Testing the model"

model.eval()
output = model(features.float(), edge_index, edge_weight)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])

print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))