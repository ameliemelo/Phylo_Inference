import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm # print progress bar 
import matplotlib.pyplot as plt # plot
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import random
import sys 
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
random.seed(113)
np.random.seed(113)
torch.manual_seed(113)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(113) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

# Global parameters

device = "cpu" # which GPU to use 
batch_size_max = 64 # max. number of trees per batch 
n_train = 900# size of training set 
n_valid = 50# size of validation set 
n_test  = 50

# Loading trees and their corresponding parameters
pandas2ri.activate()

fname_ltt = "/home/amelie/These/Phylo_Inference/data/ltt-100k-bisse.rds" #crbd ou bisse
fname_param = "/home/amelie/These/Phylo_Inference/data/true-parameters-100k-bisse.rds"

readRDS = robjects.r['readRDS']

df_ltt = readRDS(fname_ltt)
df_ltt = pandas2ri.rpy2py(df_ltt) # data.frame containing tree information
df_param = readRDS(fname_param)
df_param = pandas2ri.rpy2py(df_param) # data.frame containing target parameters
true = pandas2ri.rpy2py(df_param) 
with (robjects.default_converter + pandas2ri.converter).context():  
    true= robjects.conversion.get_conversion().rpy2py(true)

n_param = len(df_param) # number of parameters to guess for each tree 

df_ltt.fillna(0, inplace=True) 
df_ltt = np.transpose(df_ltt) 
df_ltt = pd.DataFrame(df_ltt)
n_trees = df_ltt.shape[0]

print("chargement des données")



# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 

train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 

np.save("test_indices.npy", test_ind)
# Choosing the tree indices for training, validation and test randomly 

train_inputs = torch.tensor(df_ltt.iloc[train_ind].values).float().to(device)
valid_inputs = torch.tensor(df_ltt.iloc[valid_ind].values).float().to(device)
test_inputs= torch.tensor(df_ltt.iloc[test_ind].values).float().to(device)


train_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in train_ind]).float().to(device)
valid_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in valid_ind]).float().to(device)
test_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in test_ind]).float().to(device)


train_dataset = TensorDataset(train_inputs, train_targets)
valid_dataset = TensorDataset(valid_inputs, valid_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

train_dl = DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size_max, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Build the neural network

n_input = df_ltt.shape[1]
n_out = 2
n_hidden = 8  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100  # maximum number of epochs for the training :100
patience = 5  # patience of the early stopping: normalement 3
n_layer  = 3
ker_size =5
epoch = 1
trigger = 0
last_loss = 100


class CNN(nn.Module):
    def __init__(self, n_input, n_out, n_hidden, n_layer, ker_size, p_dropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=ker_size)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=2*n_hidden, kernel_size=ker_size)
        self.conv3 = nn.Conv1d(in_channels=2*n_hidden, out_channels=4*n_hidden, kernel_size=ker_size)
        n_flatten = self.compute_dim_output_flatten_cnn(n_input, n_layer, ker_size)
        self.fc1 = nn.Linear(in_features=n_flatten * (4*n_hidden), out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=n_out)
        self.dropout = nn.Dropout(p=p_dropout)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)  
        x = F.avg_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  
        x = F.avg_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)  
        x = F.avg_pool1d(x, kernel_size=2)

        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x

    def compute_dim_output_flatten_cnn(self, n_input, n_layer, ker_size):
        for i in range(n_layer):
            n_input = (n_input - ker_size + 1) // 2
        return n_input 
    


cnn = CNN(n_input, n_out, n_hidden, n_layer, ker_size, p_dropout).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn.to(device)


learning_rate = 0.001
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = False

# If checkpoint exists, load the model and don't train it
check= True
if check == True:
    checkpoint = torch.load("checkpoints/model/bisse/CNN_LTT_checkpoint.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(checkpoint['model_state_dict'])

    cnn.eval()
    pred = [[] for _ in range(n_out)]
    true_list = [[] for _ in range(n_out)] 

    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = cnn(inputs.unsqueeze(1).to(device))
        p = output.detach().cpu().numpy() 
        for i in range(n_out):
            pred[i].extend(p[:, i])
            true_list[i].extend(targets[:, i].cpu().numpy()) 

    # Calcul des erreurs
    n = len(pred[0])
    error_qo1 = np.sum(np.abs(np.array(pred[0]) - np.array(true_list[0])))
    lambda_0 = np.sum(np.abs(np.array(pred[1]) - np.array(true_list[1])))

    print("Error q01: ", error_qo1 / n)
    print("Error lambda0: ", lambda_0 / n)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(true_list[0], pred[0], color='blue', label="lambda0")
    axs[0].plot(true_list[0], true_list[0], color='red', linestyle='--', label='Ideal line')
    axs[0].set_xlabel('True Value')
    axs[0].set_ylabel('Predicted Value')
    axs[0].set_title('Parameter lambda0')
    axs[0].legend()

    axs[1].scatter(true_list[1], pred[1], color='blue', label="q01")
    axs[1].plot(true_list[1], true_list[1], color='red', linestyle='--', label='Ideal line')
    axs[1].set_xlabel('True Value')
    axs[1].set_ylabel('Predicted Value')
    axs[1].set_title('Parameter q01')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    sys.exit()


opt = torch.optim.Adam(cnn.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

loss_fn = nn.L1Loss()
train_losses = []
valid_losses = []


# Training
def train_batch(inputs, targets):
    opt.zero_grad()
    outputs = cnn(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    opt.step()
    return loss.item()

# Validation
def valid_batch(inputs, targets):
    outputs = cnn(inputs)
    loss = loss_fn(outputs, targets)
    return loss.item()



while epoch < n_epochs and trigger < patience:
    cnn.train()
    train_loss = []
    for inputs, targets in tqdm(train_dl):
        targets = targets[:, [1, 0]]
        inputs, targets = inputs.to(device), targets.to(device)
        loss = train_batch(inputs.unsqueeze(1), targets)
        train_loss.append(loss)
    print(f"epoch {epoch}/{n_epochs} - train - loss: {np.mean(train_loss)}")
    train_losses.append(np.mean(train_loss))

    cnn.eval()
    valid_loss = []
    with torch.no_grad():
        for inputs, targets in tqdm(valid_dl):
            targets = targets[:, [1, 0]]
            inputs, targets = inputs.to(device), targets.to(device)
            loss = valid_batch(inputs.unsqueeze(1), targets)
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
plt.yscale('log')  
plt.title('Training and Validation Loss (Log Scale)')
plt.legend()
plt.show()

# Evaluation
cnn.eval()
pred = [[] for _ in range(n_out)]
with torch.no_grad():
    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = cnn(inputs.unsqueeze(1).to(device))
        p = output.cpu().numpy()
        for i in range(n_out):
            pred[i].extend(p[:, i])
        

pred = np.array(pred)

file_path = "pred_bisse_CNN_ltt.npy"

np.save(file_path, pred)     

