import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fonction import *
import pandas as pd

# Importing libraries 

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, global_mean_pool 
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
from tqdm import tqdm # print progress bar 
import pickle # save object 
import matplotlib.pyplot as plt # plot
import numpy as np
import random as rd 

from sklearn.preprocessing import StandardScaler
torch.manual_seed(113)
np.random.seed(113)

def scale_summary_statistics(df, n_taxa):
    # Identifier les colonnes à mettre à l'échelle
    col_ltt_t = [col for col in df.columns if col.startswith("ltt_t")]
    col_ltt_n = [col for col in df.columns if col.startswith("ltt_N")]
    col_ss = [col for col in df.columns if col not in col_ltt_t + col_ltt_n ]
    
    # Mettre à l'échelle les colonnes ltt_t et ltt_N
    df[col_ltt_t] /= abs(df[col_ltt_t].min())
    df[col_ltt_n] /= n_taxa
    
    # Mettre à l'échelle les autres colonnes
    scaler = StandardScaler()
    df[col_ss] = scaler.fit_transform(df[col_ss])
    
    return df


# Global parameters
load_data = True # if data is already saved, don't compute just load it
device = "cpu" # which GPU to use 
batch_size_max = 64 # max. number of trees per batch #normalement 4
n_train = 90000# size of training set 
n_valid = 5000# size of validation set 
n_test  = 5000 

# Loading trees and their corresponding parameters


pandas2ri.activate()
fname_sumstat = "data/sumstat-100k-crbd.rds"
fname_param = "data/true-parameters-100k-crbd.rds"
# fname_sumstat= "data/new-phylogeny-crbd-sumstat-1.rds"
# fname_param = "data/true-parameters-crbd-new-1.rds"
readRDS = robjects.r['readRDS']
df_sumstat = readRDS(fname_sumstat)
df_sumstat = pandas2ri.rpy2py(df_sumstat) # data.frame containing tree information
df_param = readRDS(fname_param)
true = pandas2ri.rpy2py(df_param) # data.frame containing target parameters
df_sumstat = df_sumstat.iloc[:, :-2] #enlever les true param
df_sumstat= scale_summary_statistics(df_sumstat,1000)
n_param = len(df_param) # number of parameters to guess for each tree 
n_trees = len(df_sumstat) # total number of trees of the dataset 
print(n_trees)
# Creating train, valid and test set 

# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
# rd.shuffle(ind) 
np.random.shuffle(ind) 
train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 
print(test_ind)

indices = np.array(test_ind)

# Spécifier le chemin du fichier de sauvegarde
file_ind = "test_indices.npy"



n_taxa = [100, 1000]  # range of phylogeny size

device = "cpu"  # change if you want to compute on GPUs



with (robjects.default_converter + pandas2ri.converter).context():  #To convert in pandas
    true= robjects.conversion.get_conversion().rpy2py(true)

train_inputs = torch.tensor(df_sumstat.iloc[train_ind].values).float().to(device)
valid_inputs = torch.tensor(df_sumstat.iloc[valid_ind].values).float().to(device)
test_inputs= torch.tensor(df_sumstat.iloc[test_ind].values).float().to(device)

train_targets = torch.tensor([[true['lambda'][i], true['mu'][i]] for i in train_ind]).float().to(device)
valid_targets = torch.tensor([[true['lambda'][i], true['mu'][i]] for i in valid_ind]).float().to(device)
test_targets = torch.tensor([[true['lambda'][i], true['mu'][i]] for i in test_ind]).float().to(device)

# Création des ensembles de données pour l'entraînement et la validation
train_dataset = TensorDataset(train_inputs, train_targets)
valid_dataset = TensorDataset(valid_inputs, valid_targets)
test_dataset = TensorDataset(test_inputs, test_targets)


# Chargement des ensembles de données dans DataLoader
train_dl = DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size_max, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Build the neural network

n_in = df_sumstat.shape[1]
print(n_in)  # number of neurons of the input layer = 84
n_out = len(true) #2
n_hidden = 100  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100 # maximum number of epochs for the training normalement 10
patience = 5 # patience of the early stopping normalement 4

class SS_DNN(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, p_dropout):
        super(SS_DNN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x))) 
        
        x = self.dropout(self.relu(self.fc2(x))) 
        x = self.dropout(self.relu(self.fc3(x))) 
        x = self.dropout(self.relu(self.fc4(x))) 
        x = self.fc5(x)
        return x

dnn = SS_DNN(n_in, n_out, n_hidden, p_dropout).to(device)
learning_rate = 0.001 
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = False

# Créer un optimiseur Adam avec les paramètres spécifiés
opt = optim.Adam(dnn.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

loss_fn = nn.L1Loss()


# Training
def train_batch(inputs, targets):
    opt.zero_grad()
    outputs = dnn(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    opt.step()
    return loss.item()
    

# Validation
def valid_batch(inputs, targets):
    outputs = dnn(inputs)
    loss = loss_fn(outputs, targets)
    return loss.item()



epoch = 1
trigger = 0
last_loss = 100
train_losses = []
valid_losses = []


while epoch < n_epochs and trigger < patience:
    dnn.train()
    train_loss = []
    for inputs, targets in tqdm(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        loss = train_batch(inputs, targets)
        train_loss.append(loss)
    print(f"epoch {epoch}/{n_epochs} - train - loss: {np.mean(train_loss)}")
    train_losses.append(np.mean(train_loss))


    dnn.eval()
    valid_loss = []
    with torch.no_grad():
        for inputs, targets in tqdm(valid_dl):
            inputs, targets = inputs.to(device),targets.to(device)
            loss = valid_batch(inputs, targets)
            valid_loss.append(loss)
    current_loss = np.mean(valid_loss)
    print(f"epoch {epoch}/{n_epochs} - valid - loss: {current_loss}")
    valid_losses.append(current_loss)

    if current_loss > last_loss:
        trigger += 1
    else:
        trigger = 0
        last_loss = current_loss

    epoch += 1

plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Utilisation de l'échelle logarithmique pour l'axe des y
plt.title('Training and Validation Loss (Log Scale)')
plt.legend()
plt.show()


#Evaluation

dnn.eval()
pred = [[] for _ in range(n_out)]
with torch.no_grad():
    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = dnn(inputs.to(device))
        p = output.cpu().numpy()
        for i in range(n_out):
            pred[i].extend(p[:, i])


pred_np = np.array(pred)

# Spécifier le chemin du fichier de sauvegarde
file_path = "pred_DNN_ss_crbd_mae.npy"

# Enregistrer les prédictions dans le fichier
np.save(file_path, pred_np)     

# pred_loaded = np.load("pred_DNN_ss.npy")
# print(pred_loaded[0])

true = {'lambda': true['lambda'], 'mu': true['mu']}


param_range_in = {"lambda": [0.1,1] , "mu":[0,0.9]}
plot_error_barplot_all(pred, true, param_range_in,test_ind)



# Plot predictions vs. true values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i, param_name in enumerate(true.keys()):
    plt.subplot(1, n_out, i+1)
    plt.scatter(true[param_name][test_ind], pred[i])
    plt.plot(true[param_name][test_ind], true[param_name][test_ind], color='red', linestyle='--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(param_name)
plt.tight_layout()
plt.show()


# dnn.eval()
# target = [[] for n in range(n_out)]
# pred = [[] for _ in range(n_out)]
# with torch.no_grad():
#     for inputs, targets in tqdm(train_dl):
#         inputs, targets = inputs.to(device), targets.to(device)
#         output = dnn(inputs.to(device))
#         p = output.cpu().numpy()
#         t = targets.cpu().numpy()
#         for i in range(n_out):
#             pred[i].extend(p[:, i])
#             target[i].extend(t[:, i])


# true = {'lambda': true['lambda'], 'mu': true['mu']}


# param_range_in = {"lambda": [0.1,1] , "mu":[0,0.9]}
# plt.figure(figsize=(12, 6))
# for i, param_name in enumerate(true.keys()):
#     plt.subplot(1, n_out, i+1)
#     plt.scatter(target[i], pred[i])
#     plt.plot(target[i], target[i], color='red', linestyle='--')
#     plt.xlabel('True')
#     plt.ylabel('Predicted')
#     plt.title(param_name)
# plt.tight_layout()
# plt.show()
