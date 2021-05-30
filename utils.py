import numpy as np
import torch
import scipy.sparse as sp


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import stats
import os
import time
from sklearn import preprocessing, feature_extraction, model_selection
from copy import deepcopy
from IPython.display import display, HTML
import scipy.sparse
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import networkx as nx
from torch import LongTensor
from torch import int64
from torch import tensor



def get_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i,c in enumerate(classes)}
    labels = np.array([classes_dict[i] for i in labels])
    return labels
    

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    inputs = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    idx = np.array(inputs[:, 0], dtype=np.int32)
    features = np.array(inputs[:, 1:-1], dtype=np.float32)
    labels = get_labels(inputs[:,-1])

    "Building the graph"
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    

    A_tilde = np.eye(labels.shape[0])
    for v1,v2 in edges:
        A_tilde[v1][v2]=1
        A_tilde[v2][v1]=1


    adj=normalize(A_tilde)
    # features = normalize(features)
    S=sp.csr_matrix(adj)
    edge_index , edge_weight = from_scipy_sparse_matrix(S)

    return torch.FloatTensor(adj), torch.tensor(features), torch.tensor(labels) , torch.tensor(edge_index) , edge_weight



def load_data2(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    inputs = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    idx = np.array(inputs[:, 0], dtype=np.int32)
    features = np.array(inputs[:, 1:-1], dtype=np.float32)
    labels = get_labels(inputs[:,-1])

    "Building the graph"
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    A = np.zeros((labels.shape[0],labels.shape[0]))
    D = np.zeros((labels.shape[0],labels.shape[0]))

    for v1,v2 in edges:
        A[v1][v2]=1
        A[v2][v1]=1
        D[v1][v1] +=1
        D[v2][v2] +=1  #Row sum method is quicker 

    L = D-A

    # L=normalize(L)

    features = normalize(features)

    return torch.FloatTensor(L), torch.tensor(features), torch.tensor(labels)


def normalize(mx):
    """Row-normalize matrix - Unchanged from Kipfs"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def distance_matrix(cutoff, points_1, points_2=None, output='dict'):
    
    """
    This function computes a distance matrix between points in 'points_1' and 'points_2'
    that are within 'cutoff' of each other.
    
    ***Input
    cutoff   : the distance cutoff
    points_1 : a numpy array of coordinates
    points_2 : a numpy array of coordinates
    output   : form of output.
    ***Output
            'dict': (default) is a dictionary with every pair of points as the keys
                                 and the distance as the value.
            'coo_matrix', 'dok_matrix', 'ndarray'
    """
    
    tree1 = scipy.spatial.cKDTree(points_1, leafsize=16)
    if points_2 is None:
        points_2 = points_1
    tree2 = scipy.spatial.cKDTree(points_2,leafsize=16)
    
    distances = tree1.sparse_distance_matrix(tree2, cutoff, output_type=output)
    return distances


# adj, features, labels = load_data()
# print(adj.shape)
# print(features.shape)
# print(labels.shape)

def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


def convert_sparse_matrix_to_sparse_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def load_data_adjacency(roi='BRAC4002.3c_ROI3_MRTX.', cutoff= 25, output='dict',label_output='Domain'):

    """
    Input: 
    roi - required reigon of interest: 
    Valid  ROI's are: ['BRAC3495.3f_ROI1_Cont_crop1.', 'BRAC3495.3f_ROI1_Cont_crop2.',
        'BRAC3326.4e_ROI1_Cont_crop1.', 'BRAC3438.6f_ROI1_Cont.',
        'BRAC3438.6f_ROI2_Cont.', 'BRAC3438.6f_ROI3_Cont.', 'BRAC3529.2d_ROI1_MRTX.', 'BRAC4002.3c_ROI2_MRTX_crop1.',
        'BRAC4002.3c_ROI2_MRTX_crop2.', 'BRAC4002.3c_ROI3_MRTX.',
        'BRAC3529.2b_ROI1_MRTX_crop2.', 'BRAC4002.3c_ROI1_MRTX.']
    cutoff - 
    """
    
    full2 = pd.read_csv('20210413_ContMRTX_neighbour_clustering_61clusters_DCsSeparated_TCrenamed_addedInfo copy.csv')
    full2_nandrop=full2.dropna()


    le_domain = preprocessing.LabelEncoder()

    full2_nandrop[label_output] = le_domain.fit_transform(full2_nandrop[label_output])

    loc_all_cell = full2_nandrop[['Location_Center_X','Location_Center_Y']].to_numpy()

    cols_to_drop=['X.1', 'X.2','X.3']

    output_df=full2_nandrop.drop(cols_to_drop, axis=1)


    # data frame for each roi. can write function to do this, or create a dictionary   

    output_df = output_df[(output_df.ROI_name==(roi))]
    output_df = output_df.sample(frac=1).reset_index(drop=True)
    
    loc_roi = output_df[['Location_Center_X','Location_Center_Y']].to_numpy()
    

    dist_1=distance_matrix(cutoff, loc_roi,loc_roi, 'coo_matrix')

    data , row, col = dist_1.data ,dist_1.row, dist_1.col

    def mini(x):
        if x!=0.0: 
            return 1/x 
        else: return x

    def mini2(x):
        if x!=0.0: 
            return 1
        else: return x

    vfunc = np.vectorize(mini)
    data1 = vfunc(data)
    
    n,_  = loc_roi.shape 
    adj = sp.coo_matrix((data1, (row, col)), shape=(n, n))

    cols_to_drop = ['Location_Center_X','Location_Center_Y']
    output_df=output_df.drop(cols_to_drop,axis=1)

    


    labels = output_df[label_output].to_numpy()
    features = output_df.drop(label_output,axis=1)

    # indexes_to_encode = [2, 3, 21, 22, 23, 24, 55]

    to_encode = ['cellID', 'cellType', 'Treatment', 'ROI_name', 'Filename', 'Clustername', 'clustername_tumour']

    for col in to_encode: 
        le_domain = preprocessing.LabelEncoder()
        features[col] = le_domain.fit_transform(features[col])
    features=features.to_numpy()

    print(adj.shape,features.shape,labels.shape)

    S=sp.csr_matrix(adj)

    edge_index , edge_weight = from_scipy_sparse_matrix(S)

    print("Data Loaded")

    return S, torch.tensor(features), torch.tensor(labels) , edge_index ,edge_weight




def distance_matrix(cutoff, points_1, points_2=None, output='dict'):
    
    """
    This function computes a distance matrix between points in 'points_1' and 'points_2'
    that are within 'cutoff' of each other.
    
    ***Input
    cutoff   : the distance cutoff
    points_1 : a numpy array of coordinates
    points_2 : a numpy array of coordinates
    output   : form of output.
    ***Output
            'dict': (default) is a dictionary with every pair of points as the keys
                                 and the distance as the value.
            'coo_matrix', 'dok_matrix', 'ndarray'
    """
    
    tree1 = scipy.spatial.cKDTree(points_1, leafsize=16)
    if points_2 is None:
        points_2 = points_1
    tree2 = scipy.spatial.cKDTree(points_2,leafsize=16)
    
    distances = tree1.sparse_distance_matrix(tree2, cutoff, output_type=output)
    return distances


def load_data_laplacian(roi='BRAC4002.3c_ROI3_MRTX.', cutoff= 25, output='dict',label_output='Domain',t_factor=None):

    """
    Input: 
    roi - required reigon of interest: 
    Valid  ROI's are: ['BRAC3495.3f_ROI1_Cont_crop1.', 'BRAC3495.3f_ROI1_Cont_crop2.',
        'BRAC3326.4e_ROI1_Cont_crop1.', 'BRAC3438.6f_ROI1_Cont.',
        'BRAC3438.6f_ROI2_Cont.', 'BRAC3438.6f_ROI3_Cont.', 'BRAC3529.2d_ROI1_MRTX.', 'BRAC4002.3c_ROI2_MRTX_crop1.',
        'BRAC4002.3c_ROI2_MRTX_crop2.', 'BRAC4002.3c_ROI3_MRTX.',
        'BRAC3529.2b_ROI1_MRTX_crop2.', 'BRAC4002.3c_ROI1_MRTX.']
    cutoff - 
    """
    print("Importing Data")
    
    full2 = pd.read_csv('20210413_ContMRTX_neighbour_clustering_61clusters_DCsSeparated_TCrenamed_addedInfo copy.csv')
    
    print("Imported Data")

    full2_nandrop=full2.dropna()

    le_domain = preprocessing.LabelEncoder()
    full2_nandrop[label_output] = le_domain.fit_transform(full2_nandrop[label_output])

    loc_all_cell = full2_nandrop[['Location_Center_X','Location_Center_Y']].to_numpy()

    cols_to_drop=['X.1', 'X.2','X.3']

    output_df=full2_nandrop.drop(cols_to_drop, axis=1)


    # data frame for each roi. can write function to do this, or create a dictionary   

    output_df = output_df[(output_df.ROI_name==(roi))]

    loc_roi = output_df[['Location_Center_X','Location_Center_Y']].to_numpy()
    
    dist_1=distance_matrix(cutoff, loc_roi,loc_roi, 'coo_matrix')

    data , row, col = dist_1.data ,dist_1.row, dist_1.col

    def mini(x):
        if x!=0.0: 
            return 1/x 
        else: return x

    def mini2(x):
        if x!=0.0: 
            return 1
        else: return x

    vfunc = np.vectorize(mini)
    data1 = vfunc(data)
    
    n,_  = loc_roi.shape 
    A = sp.coo_matrix((data1, (row, col)), shape=(n, n))
    L= sp.csgraph.laplacian(A)


    print('Building Laplacian Matrix')


    cols_to_drop = ['Location_Center_X','Location_Center_Y']
    output_df=output_df.drop(cols_to_drop,axis=1)

    output_df = output_df.sample(frac=1).reset_index(drop=True)


    labels = output_df[label_output].to_numpy()
    features = output_df.drop(label_output,axis=1)

    # indexes_to_encode = [2, 3, 21, 22, 23, 24, 55]

    to_encode = ['cellID', 'cellType', 'Treatment', 'ROI_name', 'Filename', 'Clustername', 'clustername_tumour']

    for col in to_encode: 
        le_domain = preprocessing.LabelEncoder()
        features[col] = le_domain.fit_transform(features[col])
    features=features.to_numpy()
    L =  convert_sparse_matrix_to_sparse_tensor(L)
    L= torch.tensor(L)
    print('Computing  exp(tL)')

    if t_factor!=None:
        L=torch.matrix_exp(t_factor*L)

    print('Computed  exp(tL)')
    S=sp.csr_matrix(L)
    edge_index , edge_weight = from_scipy_sparse_matrix(S)

    print("Data Loaded")
    return S, torch.tensor(features), torch.tensor(labels) , edge_index ,edge_weight

if __name__=='__main__':
    load_data_laplacian(t_factor=2)




