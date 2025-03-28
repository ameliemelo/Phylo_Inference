import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import StandardScaler

torch.manual_seed(113)
np.random.seed(113)


# Global parameters
device = "cpu" # which GPU to use 
batch_size_max = 64 
n_train = 900 # size of training set 
n_valid = 50 # size of validation set 
n_test  = 50 # size of test set 

# Loading trees and their corresponding parameters

pandas2ri.activate()
fname_sumstat = "/home/amelie/These/Phylo_Inference/data/sumstat-100k-crbd.rds"
fname_param = "/home/amelie/These/Phylo_Inference/data/true-parameters-100k-crbd.rds"

readRDS = robjects.r['readRDS']
df_sumstat = readRDS(fname_sumstat)
df_sumstat = pandas2ri.rpy2py(df_sumstat) # data.frame containing tree information
df_sumstat = df_sumstat.iloc[:, :-2] # remove true parameters
n_trees = len(df_sumstat) 


df_param = readRDS(fname_param)
true = pandas2ri.rpy2py(df_param) # data.frame containing target parameters
with (robjects.default_converter + pandas2ri.converter).context():  
    true= robjects.conversion.get_conversion().rpy2py(true)

n_param = len(df_param) 


# Choosing the tree indices for training, validation and test randomly 
ind = np.arange(0, n_trees) 
np.random.shuffle(ind) 
train_ind = ind[0:n_train]  
valid_ind = ind[n_train:n_train + n_valid]  
test_ind  = ind[n_train + n_valid:] 
indices = np.array(test_ind)



# Creating train, valid and test set 
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
print(n_in)
n_out = len(true) #2
n_hidden = 100  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100 # maximum number of epochs for the training normalement 10
patience = 5 # patience of the early stopping 
epoch = 1
trigger = 0
last_loss = 100

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

# Créer un optimiseur Adam avec les paramètres spécifiés
opt = optim.Adam(dnn.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

loss_fn = nn.L1Loss() #MAE Loss


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



train_losses = []
valid_losses = []

# If checkpoint exists, load the model and don't train it
check= True
if check == True:
    checkpoint = torch.load("crbd/MLP_SS_checkpoint.pth", map_location=torch.device('cpu'))
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

# Specify the path to save the predictions
file_path = "pred_crbd_MLP_ss.npy"
np.save(file_path, pred_np)     


true = {'lambda': true['lambda'], 'mu': true['mu']}



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
