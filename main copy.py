import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy, load_data_adjacency, load_data_laplacian
from utils2 import load_data_adjacency_clusters_with_greater_than_x, load_data_adjacency_n_greatest_clusters
import matplotlib.pyplot as plt
from layer import GCN

dropout=0.5
epochs=50
hidden_dim=64
lr = 0.1
weight_decay=0.0005
t_factor=1
values=[]
plot_accuracy = True
roi = 'BRAC3529.2b_ROI1_MRTX_crop2.'
label_output = 'cluster'

torch.manual_seed(0)

def main(roi='BRAC3529.2b_ROI1_MRTX_crop2.',dropout=0.3,epoch=70,hidden_dim=64,lr=0.1,weight_decay=0.0005,label_output='cluster',cutoff=40,n_clusters=15, plot_accuracy=False):
    print(f'Roi: {roi}')
    f=open(f'plots3/Training log: {roi}.txt','a')

    adj, features, labels , edge_index , edge_weight = load_data_adjacency_n_greatest_clusters(roi,cutoff=cutoff,label_output=label_output,n_clusters=n_clusters)
    n = labels.shape[0]

    values=[]

    idx_train = range((int(np.floor(0.7*n)))) 
    idx_val = range((int(np.floor(0.7*n))), (int(np.floor(0.8*n))))
    idx_test = range((int(np.floor(0.8*n))), (int(np.floor(1*n))))


    model = GCN(features.shape[1],
                hidden_dim,
                labels.max().item() + 1,
                dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    print("Training the model")
    f.write("Training the model\n")

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
        values.append(acc_train.item())
        f.write(f'Epoch: {epoch+1}, loss_train: {loss_train.item():.4f}, acc_train: {acc_train.item():.4f}, loss_val: {loss_val.item():.4f}, acc_val: {acc_val.item():.4f}, time:{time.time() - t:.4f}\n')
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

    plt.plot(values)
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.title(f"ROI: {roi}")
    plt.savefig(f'plots3/15 largest clusters - ROI: {roi}, (ELU,Sigmoid,ELU,SM), Test Accuracy: {acc_test}, Output: {label_output}.png')
    plt.clf()

    print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))

    f.write(f"Test set results: lost={loss_test.item():.4f} , accuracy= {acc_test.item():.4f}")
    f.close()
    return loss_test.item(), acc_test.item()

    if plot_accuracy:
        plt.plot(values)
        plt.xlabel('Epochs')
        plt.ylabel("Accuracy")
        plt.title(f"Epochs: {epochs}, ROI: {roi} , 3 layer GCN: (Relu,Sigmoid,log_softmax), Test Accuracy: {acc_test}, Output: {label_output}")
        plt.savefig(f'plots/20 largest clusters , Epochs: {epochs}, ROI: {roi} , 3 layer GCN: (Relu,sigmoid,log_softmax), Test Accuracy: {acc_test}, Output: {label_output}.png')
    # plt.show()




rois= ['BRAC3495.3f_ROI1_Cont_crop1.', 'BRAC3495.3f_ROI1_Cont_crop2.',
        'BRAC3326.4e_ROI1_Cont_crop1.', 'BRAC3438.6f_ROI1_Cont.',
        'BRAC3438.6f_ROI2_Cont.', 'BRAC3438.6f_ROI3_Cont.', 'BRAC3529.2d_ROI1_MRTX.', 'BRAC4002.3c_ROI2_MRTX_crop1.',
        'BRAC4002.3c_ROI2_MRTX_crop2.', 'BRAC4002.3c_ROI3_MRTX.',
        'BRAC3529.2b_ROI1_MRTX_crop2.', 'BRAC4002.3c_ROI1_MRTX.'][2:]

scores =  []
for i,roi in enumerate(rois):
    print(f'ROI {i+1} of {len(rois)}')
    loss,test = main(roi=roi)
    scores.append(test)

print(scores)

y_pos = np.arange(len(rois))
plt.bar(y_pos, scores, align='center')
plt.xticks(y_pos, rois)
plt.ylabel('Accuracy')
plt.title('Testing Accuracy of model')
plt.show()

