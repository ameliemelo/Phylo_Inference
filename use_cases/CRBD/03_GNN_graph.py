# -----------------------------------------------------------
# INFERING MACROEVOLUTIONARY RATES WITH GRAPH NEURAL NETWORKS
# -----------------------------------------------------------


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
from fonction import *
import warnings
warnings.filterwarnings("ignore")

from fonction import *

import matplotlib.pyplot as plt

torch.manual_seed(113)
np.random.seed(113)

# Global parameters
load_data =True# if data is already saved, don't compute just load it
device = "cpu" # which GPU to use 
batch_size_max = 64 # max. number of trees per batch 
n_train = 90000# size of training set 
n_valid = 5000# size of validation set 
n_test  = 5000   # size of test set 


# Loading trees and their corresponding parameters

pandas2ri.activate()
fname_graph = "data/graph-100k-crbd.rds"
fname_param = "data/true-parameters-100k-crbd.rds"
readRDS = robjects.r['readRDS']
df_graph = readRDS(fname_graph)
df_graph = pandas2ri.rpy2py(df_graph) # data.frame containing tree information

df_param = readRDS(fname_param)
df_param = pandas2ri.rpy2py(df_param) # data.frame containing target parameters
# print(len(df_graph))
with (robjects.default_converter + pandas2ri.converter).context():  #To convert in pandas
    plote= robjects.conversion.get_conversion().rpy2py(df_param)


n_param = len(df_param) # number of parameters to guess for each tree 
n_trees =len(df_graph) # total number of trees of the dataset 

# Format data 



def convert_df_to_tensor(df_node, df_edge, params):
    """
    Convert the data frames containing node and edge information 
    to a torch tensor that can be used to feed neural 
    """

    # Trier le DataFrame par la première colonne
    df_node_sorted = df_node.sort_values(by=df_node.columns[0])
    
    total_lines = df_node.shape[0]
    n = (total_lines + 1) // 2  # Garder les n premiers nœuds

    # Conserver uniquement les n premières lignes
    df_node_unique = df_node_sorted.head(n)
    max_value = df_node_unique.max().max()

    # Soustraire la plus grande valeur de chaque élément du DataFrame
    df_node_unique -= max_value

    # Créer un dictionnaire de mapping pour les nœuds
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(df_node_unique.index)}
    
    # Mettre à jour les indices des arêtes en fonction des nouveaux indices des nœuds
    l1, l2 = [], []
    for i in range(df_edge.shape[0]):
        edge = df_edge.iloc[i]
        u, v = node_id_map.get(str(int(edge[0])), -1), node_id_map.get(str(int(edge[1])), -1)
        if u != -1 and v != -1:  # Vérifier que les indices des nœuds sont valides
            l1.extend([u, v])
            l2.extend([v, u])
    
    edge_index = torch.tensor([l1, l2], dtype=torch.long)

    # Construire le tenseur des caractéristiques des nœuds
    x = torch.tensor(df_node_unique.values, dtype=torch.float)


    # Construire le tenseur des paramètres cibles
    y = torch.tensor(params, dtype=torch.float)

    # Créer un objet Data
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# fname = fname_graph[:-6] + "geomtensor" + ".obj" # file name 


fname = "data/graph-100k-crbd-distsorted-1tips-maxvalue.obj" # file name 

if (not load_data):

    data_list  = []
    print("Formating data...")
    for n in tqdm(range(n_trees)):
        df_node, df_edge = df_graph[n][0], df_graph[n][1] #0 recup node et attribut 1 recup infos edges
        with (robjects.default_converter + pandas2ri.converter).context():  #To convert in pandas
            df_edge= robjects.conversion.get_conversion().rpy2py(df_edge)
            df_node= robjects.conversion.get_conversion().rpy2py(df_node)
            columns_to_drop = ['mean.edge','time.asym', 'clade.asym', 'descendant', 'ancestor']
            df_node = df_node.drop(columns=columns_to_drop)
            
        params = [df_param[i][n] for i in range(n_param)]
        data = convert_df_to_tensor(df_node, df_edge, params)
        data_list.append(data)
    print("Formating data... Done.")
    
    file = open(fname, "wb") # file handler 
    pickle.dump(data_list, file) # save data_list
    print("Formated data saved.")

else:

    file = open(fname, "rb")
    data_list = pickle.load(file)
    print("Formated data loaded.")

print(data_list[0].x)

# Creating train, valid and test set 

# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 
train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 

# Splitting the dataset between training, validation and test. 
train_data = [data_list[i].to(device=device) for i in train_ind]
valid_data = [data_list[i].to(device=device) for i in valid_ind]
test_data  = [data_list[i].to(device=device) for i in test_ind]

# Converting the list to DataLoader
train_dl = DataLoader(train_data, batch_size = batch_size_max, shuffle = True)
valid_dl = DataLoader(valid_data, batch_size = batch_size_max, shuffle = False)
test_dl  = DataLoader(test_data , batch_size = 1)


# Creating the GNN architecture

class GCN(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.conv1 = GCNConv(n_in, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.conv4 = GCNConv(n_hidden, 2*n_hidden)
        self.lin1  = torch.nn.Linear(2*n_hidden, n_hidden)
        self.lin2  = torch.nn.Linear(n_hidden, n_out)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.001, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.001, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.001, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.001, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
import torch.nn as nn
loss_fn = nn.MSELoss()

# Defining training and validation loop 

def train(model, batch):
    optimizer.zero_grad()
    out = model(batch)
    batch_size = int(max(data.batch) + 1) # number of trees in the batch 
    target = data.y.reshape([batch_size, n_out])
    target = target[:, [1, 0]]
    loss = loss_fn(out, target)
    loss.backward() # backward propagation 
    optimizer.step()
    return(loss)

def valid(model, batch):
    out = model(batch)
    batch_size = int(max(data.batch) + 1) # number of trees in the batch 
    target = data.y.reshape([batch_size, n_out])
    target = target[:, [1, 0]]
    loss = loss_fn(out, target)
    return(loss)


# Setting up the training 
n_in = data_list[0].num_node_features #6
print(n_in)
n_out = len(data_list[0].y)
n_hidden = 50
n_epochs = 10
model = GCN(n_in, n_hidden, n_out).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
train_losses = []
valid_losses = []
# Training loop 
for epoch in range(n_epochs):

    # Training 
    model.train()
    train_loss = []
    for data in tqdm(train_dl):
        loss = train(model, data) # train model and get loss
        loss = float(loss.to(device = "cpu"))
        train_loss.append(loss)
    mean_loss = np.mean(train_loss)
    print("Epoch %d - Train Loss %.4f" % (epoch, float(mean_loss))) # print progression 
    train_losses.append(np.mean(train_loss))

    # Validation 
    model.eval()
    valid_loss = []
    for data in tqdm(valid_dl):
        loss = valid(model, data) # train model and get loss
        loss = float(loss.to(device = "cpu"))
        valid_loss.append(loss)
    mean_loss = np.mean(valid_loss)
    print("Epoch %d - Valid Loss %.4f" % (epoch, float(mean_loss))) # print progression 
    valid_losses.append(mean_loss)

# plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
# plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.yscale('log')  # Utilisation de l'échelle logarithmique pour l'axe des y
# plt.title('Training and Validation Loss (Log Scale)')
# plt.legend()
# plt.show()

#test_dl = DataLoader(data_list[:500], batch_size = 1)
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

# Spécifier le chemin du fichier de sauvegarde
file_path = "GCN_ismael_1feature_avecdonnée_modif.npy"

# Enregistrer les prédictions dans le fichier
np.save(file_path, pred_np)     

# Spécifier le chemin du fichier de sauvegarde
pred_list = np.array(pred_list)
true_list = np.array(true_list)

error_lambda = (np.sum((pred_list[0]-true_list[0])**2))/np.sum(true_list[0]**2)
print("lambda0",error_lambda)
error_mu = (np.sum((pred_list[1]-true_list[1])**2))/np.sum(true_list[1]**2)
print("q",error_mu)

prediction = [[], []]
prediction[0] = pred_list[0]
prediction[1] = pred_list[1]

true= [[], []]
true[0] = true_list[0]
true[1] = true_list[1]


# plot_error_barplot_all(prediction, true, param_range_in, test_ind)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Parcourir les deux paramètres
# Plot prédit vs vrai pour le paramètre i
axs[0].scatter(true[0], prediction[0], color='blue', label="lambda0")
axs[0].plot(true[0], true[0], color='red', linestyle='--', label='Ideal line')
axs[0].set_xlabel('True Value')
axs[0].set_ylabel('Predicted Value')
axs[0].set_title('Parameter lambda0')
axs[0].legend()

axs[1].scatter(true[1], prediction[1], color='blue', label="q01")
axs[1].plot(true[1], true[1], color='red', linestyle='--', label='Ideal line')
axs[1].set_xlabel('True Value')
axs[1].set_ylabel('Predicted Value')
axs[1].set_title('Parameter q01')
axs[1].legend()

plt.tight_layout()
plt.show()


###########----------------------------------------------------------------------------------------------------------------------------------











# pred_loaded = np.load("pred_DNN_ss.npy")
# print(pred_loaded[0])
# Créer deux sous-graphiques
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Parcourir les deux paramètres
# for i in range(n_param):
#     # Plot prédit vs vrai pour le paramètre i
#     axs[i].scatter(true_list[i], pred_list[i], color='blue', label=f'Predicted vs True ({df_param.names[i]})')
#     axs[i].plot(true_list[i], true_list[i], color='red', linestyle='--', label='Ideal line')
#     axs[i].set_xlabel('True Value')
#     axs[i].set_ylabel('Predicted Value')
#     axs[i].set_title(f'Parameter {df_param.names[i]}')
#     axs[i].legend()

# plt.tight_layout()
# plt.show()

# true = {'lambda': plote['lambda'], 'mu': plote['mu']}
# param_range_in = {"lambda": [0.1,1] , "mu":[0,0.9]}
# # plot_error_barplot_all(pred_list, true, param_range_in,test_ind)




# pred_list, true_list = [[] for n in range(n_param)], [[] for n in range(n_param)]
# model.eval()
# for data in train_dl:
#     batch_size = int(max(data.batch) + 1)
#     out = model(data.to(device=device))
#     pred_params = out.tolist()
#     true_params = data.y.reshape([batch_size, n_out]).tolist()
#     for i in range(n_param):
#         pred_list[i].extend([param[i] for param in pred_params]) 
#         true_list[i].extend([param[i] for param in true_params])
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Parcourir les deux paramètres
# for i in range(n_param):
#     # Plot prédit vs vrai pour le paramètre i
#     axs[i].scatter(true_list[i], pred_list[i], color='blue', label=f'Predicted vs True ({df_param.names[i]})')
#     axs[i].plot(true_list[i], true_list[i], color='red', linestyle='--', label='Ideal line')
#     axs[i].set_xlabel('True Value')
#     axs[i].set_ylabel('Predicted Value')
#     axs[i].set_title(f'Parameter {df_param.names[i]}')
#     axs[i].legend()

# plt.tight_layout()
# plt.show()