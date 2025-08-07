import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
#import rpy2.robjects as robjects # load R object 
#from rpy2.robjects import pandas2ri # load R object 
import matplotlib.pyplot as plt # plot
import sys
from tqdm import tqdm # print progress bar  
import numpy as np
import pandas as pd
import random
random.seed(113)
np.random.seed(113)
torch.manual_seed(113)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(113) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

torch.manual_seed(113)
np.random.seed(113)  # For reproducibility

# Parameters
batch_size_max = 64  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_path = "/data/"  
file_names = [
    "cblv-100k-bisse1.csv",
    "cblv-100k-bisse2.csv",
    "cblv-100k-bisse3.csv",
    "cblv-100k-bisse4.csv",
    "cblv-100k-bisse5.csv",
    "cblv-100k-bisse6.csv",
    "cblv-100k-bisse7.csv",
    "cblv-100k-bisse8.csv",
    "cblv-100k-bisse9.csv",
    "cblv-100k-bisse10.csv"
]
file_paths = [base_path + file_name for file_name in file_names]

#readRDS = robjects.r['readRDS']
tensor_list = []
for file_path in file_paths:
    df_cblv = pd.read_csv(file_path)
    data_tensor = torch.tensor(df_cblv.values).float()
    tensor_list.append(data_tensor)

df_cblv = torch.cat(tensor_list, dim=0)
print(f"Taille totale du dataset concaténé : {df_cblv.shape}")

# Now for the true parameters
param_base_path =  "/data/"
param_file_names = [
    "true-parameters-100k-bisse1.csv",
    "true-parameters-100k-bisse2.csv",
    "true-parameters-100k-bisse3.csv",
    "true-parameters-100k-bisse4.csv",
    "true-parameters-100k-bisse5.csv",
    "true-parameters-100k-bisse6.csv",
    "true-parameters-100k-bisse7.csv",
    "true-parameters-100k-bisse8.csv",
    "true-parameters-100k-bisse9.csv",
    "true-parameters-100k-bisse10.csv"
]
param_file_paths = [param_base_path + file_name for file_name in param_file_names]

# Reading and concatenating parameter files
all_true_params = []
for file_path in param_file_paths:
    df_param = pd.read_csv(file_path)  # Read CSV for parameters
    df_param = torch.tensor(df_param.values).float()
    all_true_params.append(df_param)


# Concatenate parameter dataframes into one
true = torch.concat(all_true_params, dim=0)
print(f"Taille totale du DataFrame de paramètres concaténés : {true.shape}")



# Randomly shuffle indices for the train, valid, test splits
n_total = df_cblv.shape[0]
ind = np.arange(0, n_total)
np.random.shuffle(ind)

n_train = int(0.9 * n_total)  # 90% for training
n_valid = int(0.05 * n_total) # 5% for validation
n_test  = n_total - n_train - n_valid # 5% for testing
print("Number of tree", n_total)

train_ind = ind[0:n_train]
valid_ind = ind[n_train:n_train + n_valid]
test_ind = ind[n_train + n_valid:]


train_inputs = df_cblv[train_ind]
valid_inputs = df_cblv[valid_ind]
test_inputs = df_cblv[test_ind]
print("Taille du jeu de données d'entraînement :", train_inputs.shape)
print("Taille du jeu de données de validation :", valid_inputs.shape)
print("Taille du jeu de données de test :", test_inputs.shape)

lambda0_col = 0  # Indice de la colonne 'lambda0' dans le tensor (ajustez si nécessaire)
q01_col = 4

train_targets = true[train_ind][:, [lambda0_col, q01_col]]
valid_targets = true[valid_ind][:, [lambda0_col, q01_col]]
test_targets = true[test_ind][:, [lambda0_col, q01_col]]

# Création des ensembles de données pour l'entraînement et la validation
train_dataset = TensorDataset(train_inputs, train_targets)
valid_dataset = TensorDataset(valid_inputs, valid_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

# Chargement des ensembles de données dans DataLoader
train_dl = DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size_max, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Build the neural network

n_input = df_cblv.shape[1]
n_out = 2 
n_hidden = 8  # number of neurons in the hidden layers
p_dropout = 0.01  # dropout probability
n_epochs = 100  # maximum number of epochs for the training :100
patience = 5  # patience of the early stopping
n_layer  = 4
ker_size =10
epoch = 1
trigger = 0
last_loss = 100


class CNN(nn.Module):
    def __init__(self, n_input, n_out, n_hidden, n_layer, ker_size, p_dropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=ker_size)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=2*n_hidden, kernel_size=ker_size)
        self.conv3 = nn.Conv1d(in_channels=2*n_hidden, out_channels=4*n_hidden, kernel_size=ker_size)
        self.conv4 = nn.Conv1d(in_channels=4*n_hidden, out_channels=8*n_hidden, kernel_size=ker_size)
        n_flatten = self.compute_dim_output_flatten_cnn(n_input, n_layer, ker_size)
        self.fc1 = nn.Linear(in_features=n_flatten * (8*n_hidden), out_features=100)
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

        x = F.relu(self.conv4(x))
        x = self.dropout(x) 
        x = F.avg_pool1d(x, 2)
        
        x = x.view(x.size(0), -1)
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
num_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(f"Number of parameters in the model : {num_params}")
learning_rate = 0.0005
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = False

# If checkpoint exists, load the model and don't train it
check= True
if check == True:
    checkpoint = torch.load("/checkpoints/CNN_CBLV_checkpoint.pth", map_location=device)
    cnn.load_state_dict(checkpoint['model_state_dict'])

    cnn.eval()
    pred = [[] for _ in range(n_out)]
    true_list = [[] for _ in range(n_out)] 

    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = output = cnn(inputs.unsqueeze(1).to(device))
        p = output.detach().cpu().numpy() 
        for i in range(n_out):
            pred[i].extend(p[:, i])
            true_list[i].extend(targets[:, i].cpu().numpy()) 

    # Calcul des erreurs
    n = len(pred[0])
    error_lambda_0 = np.sum(np.abs(np.array(pred[0]) - np.array(true_list[0])))
    error_q01 = np.sum(np.abs(np.array(pred[1]) - np.array(true_list[1])))

    print("Error lambda0: ", error_lambda_0 / n)
    print("Error q01: ", error_q01 / n)

    np.save("/data/pred_bisse_CNN_cblv.npy", pred)

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




opt = optim.Adam(cnn.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

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
        inputs, targets = inputs.to(device), targets.to(device)
        loss = train_batch(inputs.unsqueeze(1), targets)
        train_loss.append(loss)
    print(f"epoch {epoch}/{n_epochs} - train - loss: {np.mean(train_loss)}")
    train_losses.append(np.mean(train_loss))

    cnn.eval()
    valid_loss = []
    with torch.no_grad():
        for inputs, targets in tqdm(valid_dl):
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

# Evaluation
cnn.eval()
pred, true_list = [[] for _ in range(n_out)], [[] for _ in range(n_out)]
with torch.no_grad():
    for inputs, targets in tqdm(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        output = cnn(inputs.unsqueeze(1).to(device))
        p = output.cpu().numpy()

        for i in range(n_out):
            pred[i].extend(p[:, i])
            true_list[i].extend(targets[:, i])
        



pred_np = np.array(pred)
file_path = "pred_bisse_CNN_cblv.npy"
np.save(file_path, pred_np)  
