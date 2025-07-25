import sys
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
from tqdm import tqdm # print progress bar 
import pickle # save object 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch_scatter import scatter_add
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
load_data =False # if data is already saved on the good format, else False
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
batch_size_max = 64 # max. number of trees per batch 
n_train = 9000 # size of training set 
n_valid = 5000 # size of validation set 
n_test  = 5000

pandas2ri.activate()
fname_graph = "/home/amelie/These/Phylo_Inference/data/graph-100k-crbd.rds"
fname_param = "/home/amelie/These/Phylo_Inference/data/true-parameters-100k-crbd.rds"
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

    x = torch.tensor(df_node_sorted.values, dtype=torch.float)
    y = torch.tensor(params, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


fname="/home/amelie/These/Phylo_Inference/data/graph-100k-crbd_dist_tips_sorted_maxvalue_geomtensor.obj"


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
        selected_indices = [0, 1]  # drop columns that are not needed
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



def get_valid_node_indices(initial_num_nodes):
    num_conv_layers =3
    pooling_factor = 2
    valid_node_count = initial_num_nodes
    for _ in range(num_conv_layers):
        valid_node_count = (valid_node_count) // pooling_factor

    return valid_node_count


def to_dense_batch(x, batch=None, fill_value=0, max_num_nodes=2000):
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    batch_size = batch[-1].item() + 1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = torch.as_tensor(fill_value, device=x.device)
    out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    return out, num_nodes


class GNN(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, ker_size, p_dropout):
        super(GNN, self).__init__()
        self.mp1 = GCNConv(n_in, n_hidden)
        self.mp2 = GCNConv(n_hidden, n_hidden)
        self.n_parts=10
        self.conv1 = nn.Conv1d(in_channels=n_hidden, out_channels=2*n_hidden, kernel_size=ker_size, padding = "same")
        self.conv2 = nn.Conv1d(in_channels=2*n_hidden, out_channels=4*n_hidden, kernel_size=ker_size, padding="same")
        self.conv3 = nn.Conv1d(in_channels=4*n_hidden, out_channels=8*n_hidden, kernel_size=ker_size, padding = "same")
        self.fc1 = nn.Linear(in_features=8*n_hidden*self.n_parts, out_features=100)
        self.fc2 = nn.Linear(100, n_out)
        self.dropout = nn.Dropout(p=p_dropout)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = data.batch.max().item() + 1

        # Message Passing layers
        x = self.mp1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.mp2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Padding
        x, num_nodes = to_dense_batch(x, batch)
        x_padded = x.permute(0, 2, 1)

        # Convolutional layers
        x_padded = self.conv1(x_padded)
        x_padded = F.relu(x_padded)
        x_padded = self.dropout(x_padded)
        x_padded = F.avg_pool1d(x_padded, kernel_size=2)

        x_padded = F.relu(self.conv2(x_padded))
        x_padded = self.dropout(x_padded)
        x_padded = F.avg_pool1d(x_padded, kernel_size=2)

        x_padded = F.relu(self.conv3(x_padded))
        x_padded = self.dropout(x_padded)
        x_padded = F.avg_pool1d(x_padded, kernel_size=2)

        # Calculate the valid nodes for each graph (without padding)
        valid_nodes = [get_valid_node_indices(n.item()) for n in num_nodes]

        selected_nodes_list = []
        for i, valid in enumerate(valid_nodes):
            valid_indices = torch.arange(valid)  # Generate valid node indices

            base_size = valid // self.n_parts  # Base size of each part
            remainder = valid % self.n_parts   # Number of remaining indices to distribute

            # Calculate the sizes of the parts
            part_sizes = [base_size + 1 if j < remainder else base_size for j in range(self.n_parts)]
            part_means = []
            start_idx = 0

            for part_size in part_sizes:
                end_idx = start_idx + part_size
                part_indices = valid_indices[start_idx:end_idx]

                start_idx = end_idx

                part_mean = x_padded[i, :, part_indices].mean(dim=1)
                part_means.append(part_mean)

            selected_nodes_list.append(torch.stack(part_means))


        selected_nodes = torch.stack(selected_nodes_list)
        selected_nodes = selected_nodes.permute(0, 2, 1)

        # Flatten and pass through final fully connected layers
        selected_nodes_flattened = selected_nodes.reshape(batch_size, -1)

        out = F.relu(self.fc1(selected_nodes_flattened))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


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

n_hidden = 8  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100  # maximum number of epochs for the training :100
patience = 5  # patience of the early stopping: normalement 3
n_layer  = 3
ker_size =5
epoch = 1
trigger = 0
last_loss = 100


n_in = data_list[0].num_node_features 
n_out = len(data_list[0].y)
model = GNN(n_in, n_out, n_hidden, ker_size, p_dropout).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
train_losses = []
valid_losses = []

# If checkpoint exists, load the model and don't train it
check= True
if check == True:
    checkpoint = torch.load("crbd/GNN_PhyloPool_checkpoint.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

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

    n = len(pred_list[0])
    error_qo1 = np.sum(np.abs(np.array(pred_list[0]) - np.array(true_list[1])))
    lambda_0 = np.sum(np.abs(np.array(pred_list[1]) - np.array(true_list[0])))
    print("Error q01: ", error_qo1/n)
    print("Error lambda0: ", lambda_0/n)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(true_list[0], pred_list[1], color='blue', label="lambda0")
    axs[0].plot(true_list[0], true_list[0], color='red', linestyle='--', label='Ideal line')
    axs[0].set_xlabel('True Value')
    axs[0].set_ylabel('Predicted Value')
    axs[0].set_title('Parameter lambda0')
    axs[0].legend()

    axs[1].scatter(true_list[1], pred_list[0], color='blue', label="q01")
    axs[1].plot(true_list[1], true_list[1], color='red', linestyle='--', label='Ideal line')
    axs[1].set_xlabel('True Value')
    axs[1].set_ylabel('Predicted Value')
    axs[1].set_title('Parameter q01')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    sys.exit()
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


    if current_loss > last_loss:
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

file_path = "pred_crbd_GNN_PhyloPool.npy"

np.save(file_path, pred_np)     
