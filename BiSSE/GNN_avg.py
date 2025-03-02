import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool 
import torch_geometric
import pickle
import numpy as np
from tqdm import tqdm
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
import warnings
warnings.filterwarnings("ignore")
import time
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random

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

load_data =False# if data is already saved on the good format, else False
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
batch_size_max = 64 
n_train = 900 # size of training set 
n_valid = 50 # size of validation set 
n_test  = 50 # size of test set 

pandas2ri.activate()
fname_graph = "/home/amelie/These/Phylo_Inference/data/graph-100k-bisse.rds"
fname_param = "/home/amelie/These/Phylo_Inference/data/true-parameters-100k-bisse.rds"
readRDS = robjects.r['readRDS']
df_graph = readRDS(fname_graph)
df_graph = pandas2ri.rpy2py(df_graph) # data.frame containing tree information

df_param = readRDS(fname_param)
df_param = pandas2ri.rpy2py(df_param) # data.frame containing target parameters


n_param = len(df_param) # number of parameters to guess for each tree 
n_trees =len(df_graph) # total number of trees of the dataset 

# Format data 

def convert_df_to_tensor(df_node, df_edge, params):

    """
    Convert the data frames containing node and edge information 
    to a torch tensor that can be used to feed neural 
    """
    # Sort the nodes by their indices
    df_node_sorted = df_node.sort_values(by=df_node.columns[0])
    n_node, n_edge = df_node_sorted.shape[0], df_edge.shape[0]
    l1, l2 = [], []

    # Update edge indices based on the new node order
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(df_node_sorted.index)}
    for i in range(n_edge):
        edge = df_edge.iloc[i]
        u, v = node_id_map[str(int(edge[0]))], node_id_map[str(int(edge[1]))]
        l1.extend([u, v])
        l2.extend([v, u])   
    edge_index = torch.tensor([l1, l2], dtype=torch.long)
    max_value = df_node_sorted['dist'].max().max()

    # Subtract the maximum value from each element in the DataFrame
    df_node_sorted['dist'] -= max_value

    tolerance = 1e-9
    df_node_sorted['dist'] = df_node_sorted['dist'].apply(lambda x: 0 if np.abs(x) < tolerance else x)

    # Replace values in 'state' column: -1 -> 0 and 0 -> -1
    df_node_sorted['state'] = df_node_sorted['state'].replace({-1: 0, 0: -1})

    x = torch.tensor(df_node_sorted.values, dtype=torch.float)
    y = torch.tensor(params, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

fname="/home/amelie/These/Phylo_Inference/data/graph-100k-bisse_dist_tips_sorted_maxvalue_geomtensor.obj"




if (not load_data):

    data_list  = []
    print("Formating data...")
    for n in tqdm(range(n_trees)):
        df_node, df_edge = df_graph[n][0], df_graph[n][1] # get the node and edge information 
        with (robjects.default_converter + pandas2ri.converter).context(): 
            df_edge= robjects.conversion.get_conversion().rpy2py(df_edge)
            df_node= robjects.conversion.get_conversion().rpy2py(df_node)
            columns_to_drop = ['mean.edge','time.asym', 'clade.asym', 'descendant', 'ancestor']
            df_node = df_node.drop(columns=columns_to_drop)
        selected_indices = [0, 4]  # 0 and 1 for crbd and 0 and 4 for bisse
        params = [df_param[i][n] for i in selected_indices]
        data = convert_df_to_tensor(df_node, df_edge, params)
        data_list.append(data)
    print("Formating data... Done.")

    file = open(fname, "wb") # file handler 
    pickle.dump(data_list, file) # save data_list
    print("Formated data saved.")

else:

    file = open(fname, "rb")
    data_list = pickle.load(file)
print(data_list[0].x)

# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 

train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 

# Splitting the dataset between training, validation and test

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
    loss.backward() 
    optimizer.step()
    return(loss)

def valid(model, data):
    out = model(data)
    batch_size = data.batch.max().item() + 1 # number of trees in the batch 
    target = data.y.reshape([batch_size, n_out])
    loss = loss_fn(out, target)
    return(loss)


# Setting up the training 
n_hidden = 50  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100  # maximum number of epochs for the training :100
patience = 5  # patience of the early stopping: normalement 3
n_layer  = 3
ker_size =5
epoch = 1
trigger = 0
last_loss = 1000

n_in = data_list[0].num_node_features 
n_out = len(data_list[0].y)
model = GCN(n_in, n_out, n_hidden, p_dropout).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
train_losses = []
valid_losses = []

# Training loop 

while epoch < n_epochs and trigger <= patience:
    start_time = time.time()

    # Training 
    model.train()
    train_loss = []
    for step, data in enumerate(train_dl):
        data = data.to(device=device)
        loss = train(model, data)
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

file_path = "pred_bisse_GNN_avg.npy"

np.save(file_path, pred_np)     


