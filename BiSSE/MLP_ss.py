import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
from tqdm import tqdm # print progress bar 
import matplotlib.pyplot as plt # plot
import numpy as np
torch.manual_seed(113)
np.random.seed(113)
import sys 


print("Chargement des données")

base_path = "/home/amelie/These/Phylo_Inference/data/" 
file_names = [
    "sumstat-100k-bisse.rds"
]
file_paths = [base_path + file_name for file_name in file_names]

device = "cuda" if torch.cuda.is_available() else "cpu"

readRDS = robjects.r['readRDS']
all_dfs = []

for file_path in file_paths:
    df_sumstat  = readRDS(file_path) 
    df_sumstat =pandas2ri.rpy2py(df_sumstat)
    df_sumstat = df_sumstat.iloc[:, :-7].join(df_sumstat.iloc[:, -1:]) # Remove parameters for training
    all_dfs.append(df_sumstat)

# Concatenation of all dataframes
df_sumstat = pd.concat(all_dfs, axis=0)
print(f"Taille totale du dataset concaténé : {df_sumstat.shape}")
n_trees = len(df_sumstat)

# Division of the dataset into training, validation and test sets
n_train = int(0.9* n_trees )
n_valid = int(0.05 * n_trees )
n_test = n_trees  - n_train - n_valid


# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 

train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 


param_base_path = "/home/amelie/These/Phylo_Inference/data/"
param_file_names = [
    "true-parameters-100k-bisse.rds"
]

param_file_paths = [param_base_path + file_name for file_name in param_file_names]

readRDS = robjects.r['readRDS']
all_true_params = []

print("Chargement des fichiers de paramètres...")
for i, file_path in enumerate(param_file_paths):
    with (robjects.default_converter + pandas2ri.converter).context():
        df_param = readRDS(file_path)  
        df_param = robjects.conversion.get_conversion().rpy2py(df_param)  
    if not isinstance(df_param, pd.DataFrame):
        df_param = pd.DataFrame(df_param)
    
    all_true_params.append(df_param)


true = pd.concat(all_true_params, axis=0, ignore_index=True)
print(f"Taille totale du DataFrame de paramètres concaténés : {true.shape}")

batch_size_max = 64

train_inputs = torch.tensor(df_sumstat.iloc[train_ind].values).float().to(device)
valid_inputs = torch.tensor(df_sumstat.iloc[valid_ind].values).float().to(device)
test_inputs= torch.tensor(df_sumstat.iloc[test_ind].values).float().to(device)

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

n_in = df_sumstat.shape[1]
n_out = 2
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
num_params = sum(p.numel() for p in dnn.parameters() if p.requires_grad)
print(f"Number of parameters in the model : {num_params}")

learning_rate = 0.001 
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = False

opt = optim.Adam(dnn.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

loss_fn = nn.L1Loss()

# If checkpoint exists, load the model and don't train it
check= True
if check == True:
    checkpoint = torch.load("checkpoints/model/bisse/MLP_SS_checkpoint.pth", map_location=torch.device('cpu'))
    dnn.load_state_dict(checkpoint['model_state_dict'])

    dnn.eval()
    pred = [[] for _ in range(n_out)]
    true_list = [[] for _ in range(n_out)] 

    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = dnn(inputs.to(device))
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
plt.yscale('log') 
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

file_path = "pred_bisse_MPL_ss.npy"

np.save(file_path, pred_np)     
