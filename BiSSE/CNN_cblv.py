import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import rpy2.robjects as robjects # load R object 
from rpy2.robjects import pandas2ri # load R object 
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


base_path = "/home/amelie/These/Phylo_Inference/data/"  
file_names = [
    "cblv-100k-bisse.rds"
]
file_paths = [base_path + file_name for file_name in file_names]

readRDS = robjects.r['readRDS']
all_dfs = []

# Concatenate all data files
for file_path in file_paths:
    df_cblv = readRDS(file_path)
    df_cblv = np.transpose(df_cblv)  
    df_cblv = pd.DataFrame(df_cblv)
    all_dfs.append(df_cblv)

df_cblv = pd.concat(all_dfs, axis=0)
print(f"Taille totale du dataset concaténé : {df_cblv.shape}")

# Now for the true parameters
param_base_path =  "/home/amelie/These/Phylo_Inference/data/" 
param_file_names = [
    "true-parameters-100k-bisse.rds"
]
param_file_paths = [param_base_path + file_name for file_name in param_file_names]

# Reading and concatenating parameter files
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


train_inputs = torch.tensor(df_cblv.iloc[train_ind].values).float().to(device)
valid_inputs = torch.tensor(df_cblv.iloc[valid_ind].values).float().to(device)
test_inputs= torch.tensor(df_cblv.iloc[test_ind].values).float().to(device)


train_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in train_ind]).float().to(device)
valid_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in valid_ind]).float().to(device)
test_targets = torch.tensor([[true['lambda0'][i], true['q01'][i]] for i in test_ind]).float().to(device)

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
    checkpoint = torch.load("checkpoints/epoch16_2.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(checkpoint['model_state_dict'])

    # checkpoint = torch.load("checkpoints/epoch16_2.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    n_param = 2
    pred_list, true_list = [[] for n in range(n_param)], [[] for n in range(n_param)]
    cnn.eval()
    for data in test_dl:
        out = model(data.to(device=device))
        pred_params = out.tolist()[0]
        true_params = data.y.tolist()
        for n in range(2):
            pred_list[n].append(pred_params[n])
            true_list[n].append(true_params[n])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Parcourir les deux paramètres
    # Plot prédit vs vrai pour le paramètre i
    axs[0].scatter(true_list[0], pred_list[0], color='blue', label="lambda0")
    axs[0].plot(true_list[0], true_list[0], color='red', linestyle='--', label='Ideal line')
    axs[0].set_xlabel('True Value')
    axs[0].set_ylabel('Predicted Value')
    axs[0].set_title('Parameter lambda0')
    axs[0].legend()

    axs[1].scatter(true_list[1], pred_list[1], color='blue', label="q01")
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
