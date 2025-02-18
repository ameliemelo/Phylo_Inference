import torch
import csv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, global_mean_pool  # load R object 
import torch_geometric
from tqdm import tqdm # print progress bar 
import pickle # save object 
import matplotlib.pyplot as plt # plot
import numpy as np
import random as rd 
from fonction import *
import warnings
warnings.filterwarnings("ignore")
from torch_scatter import scatter_add

import matplotlib.pyplot as plt
import time
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import sys

random.seed(113)
np.random.seed(113)
torch.manual_seed(113)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(113) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(113)

torch.use_deterministic_algorithms(True)

# Global parameters
load_data =True# if data is already saved, don't compute just load it
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
batch_size_max = 64 # max. number of trees per batch 
n_train = 90000# size of training set 
n_valid = 5000# size of validation set 
n_test  = 5000


# Loading trees and their corresponding parameters

# fname_graph = "data/graph-100k-bisse.rds"
fname_param = "data/true-parameters-100k-crbd.rds"

n_trees = 100000 # total number of trees of the dataset 

fname="/gpfswork/rech/hvr/uhd88jk/overfit/Reproduction_resultat/data/graph-100k-cr_dist_tips_sorted_maxvalue_geomtensor.obj"


file = open(fname, "rb")
data_list = pickle.load(file)
print(data_list[0].x)

# Creating train, valid and test set 

# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 

train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 
print(test_ind)
# Splitting the dataset between training, validation and test. 
train_data = [data_list[i].to(device=device) for i in train_ind]
valid_data = [data_list[i].to(device=device) for i in valid_ind]
test_data  = [data_list[i].to(device=device) for i in test_ind]

# Converting the list to DataLoader
train_dl = DataLoader(train_data, batch_size = batch_size_max, shuffle = True)
valid_dl = DataLoader(valid_data, batch_size = batch_size_max, shuffle = False)
test_dl  = DataLoader(test_data , batch_size = 1)


class GCN(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(n_in, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.conv4 = GCNConv(n_hidden, 2*n_hidden)
        self.fc1  = torch.nn.Linear(2*n_hidden, n_hidden)
        self.fc2  = torch.nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(p=p_dropout) # Utilisez nn.Dropout au lieu de p_dropout


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        batch_size = data.batch.max().item() + 1

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

import torch.nn as nn
# loss_fn = nn.MSELoss()

class WeightedLoss(nn.Module):
    def __init__(self, weight1, weight2):
        super(WeightedLoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, outputs, targets):
        output1, output2 = outputs[:, 0], outputs[:, 1]
        target1, target2 = targets[:, 0], targets[:, 1]

        loss1 = nn.L1Loss()(output1, target1)
        loss2 = nn.L1Loss()(output2, target2)

        weighted_loss = self.weight1 * loss1 + self.weight2 * loss2
        return weighted_loss

loss_fn = WeightedLoss(1,1)


# Defining training and validation loop 

def train(model, data):
    optimizer.zero_grad()
    out = model(data)
    batch_size = data.batch.max().item() + 1 # number of trees in the batch 
    target = data.y.reshape([batch_size, n_out])
    loss = loss_fn(out, target)
    loss.backward() # backward propagation 
    optimizer.step()
    return(loss)

def valid(model, data):
    out = model(data)
    batch_size = data.batch.max().item() + 1 # number of trees in the batch 
    target = data.y.reshape([batch_size, n_out])
    loss = loss_fn(out, target)
    return(loss)


# Setting up the training 
n_hidden = 16  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
# n_epochs = 100  # maximum number of epochs for the training :100
patience = 5  # patience of the early stopping: normalement 3
n_layer  = 3
ker_size =5
epoch = 1
trigger = 0
last_loss = 1000

n_in = data_list[0].num_node_features #6

n_out = len(data_list[0].y)
n_epochs = 100
model = GCN(n_in, n_out, n_hidden, p_dropout).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
train_losses = []
valid_losses = []
losses_every_100_batches = []
model_name = "GNN_ism"

# Training loop 

while epoch < n_epochs and trigger <= patience:
    start_time = time.time()

    # Training 
    model.train()
    train_loss = []
    for step, data in enumerate(train_dl):
        data = data.to(device=device)
        loss = train(model, data) # train model and get loss
        loss = float(loss.to(device = "cpu"))
        train_loss.append(loss)
    
    mean_loss = np.mean(train_loss)
    print("Epoch %d - Train Loss %.4f" % (epoch, float(mean_loss)),  flush=True) # print progression 
    train_losses.append(np.mean(train_loss))

    # Validation 
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for data in valid_dl:
            data = data.to(device=device)
            loss = valid(model, data) # train model and get loss
            loss = float(loss.to(device = "cpu"))
            valid_loss.append(loss)
    current_loss = np.mean(valid_loss)
    print("Epoch %d - Valid Loss %.4f" % (epoch, float(current_loss)), flush=True) # print progression 
    valid_losses.append(current_loss)


    end_time = time.time()  # End timer at the end of the epoch
    epoch_duration = end_time - start_time
    print("Epoch %d - Duration: %.2f seconds" % (epoch, epoch_duration), flush=True)  # Print epoch duration


    if current_loss >= last_loss:
        trigger += 1
    else:
        trigger = 0
        last_loss = current_loss

    epoch += 1
n_param = 2
pred_list, true_list = [[] for n in range(n_param)], [[] for n in range(n_param)]
model.eval()
for data in test_dl:
    out = model(data.to(device=device))
    pred_params = out.tolist()[0]
    true_params = data.y.tolist()
    for n in range(2):
        pred_list[n].append(pred_params[n])
        true_list[n].append(true_params[n])

pred_np = np.array(pred_list)
print(pred_np)

# Spécifier le chemin du fichier de sauvegarde
file_path = "/gpfswork/rech/hvr/uhd88jk/overfit/Reproduction_resultat/results/CRBD_gcn/gcn_ismael_mae_hidden16.npy"

# Enregistrer les prédictions dans le fichier
np.save(file_path, pred_np)     


pred_list = np.array(pred_list)
true_list = np.array(true_list)

error_lambda = (np.sum((pred_list[0]-true_list[0])**2))/np.sum(true_list[0]**2)
print("lambda0",error_lambda)
error_mu = (np.sum((pred_list[1]-true_list[1])**2))/np.sum(true_list[1]**2)
print("mu",error_mu)
