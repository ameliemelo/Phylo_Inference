import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

torch.manual_seed(113)
np.random.seed(113)
true= np.load("/BiSSE/results/true_bisse.npy")
print(true)
print(len(true[0]))


import numpy as np
import json
pred_ss = np.load("/BiSSE/results/pred_bisse_MLP_ss.npy")

prediction_ss =[[],[]]
prediction_ss[1] = pred_ss[1]
prediction_ss[0] = pred_ss[0]

pred_ltt = np.load("/BiSSE/results/pred_bisse_CNN_ltt.npy")

prediction_ltt  =[[],[]]
prediction_ltt[1] = pred_ltt[1]
prediction_ltt[0] = pred_ltt[0]

pred_gnn_PP = np.load("/BiSSE/results/pred_bisse_GNN_PhyloPool2.npy")


prediction_gnn_PP =[[],[]]
prediction_gnn_PP[1] = pred_gnn_PP[1]
prediction_gnn_PP[0] = pred_gnn_PP[0]

pred_gnn_ism = np.load("/BiSSE/results/pred_bisse_GNN_avg.npy")

prediction_gnn_ism =[[],[]]
prediction_gnn_ism[1] = pred_gnn_ism[1]
prediction_gnn_ism[0] = pred_gnn_ism[0]

pred_cblv = np.load("/BiSSE/results/pred_bisse_CNN_cblv.npy")

prediction_cblv  =[[],[]]
prediction_cblv[1] = pred_cblv[1]
prediction_cblv[0] = pred_cblv[0]

with open('/BiSSE/results/dataMLE_bisse.json', 'r') as file:
    data = json.load(file)

pred = data['pred']
true_mle = {
    'lambda0': np.array(data['true']['lambda0']),
    'q01': np.array(data['true']['q01'])
}



prediction_models = [
    {"name": "MLP-SS", "pred_lambda0": prediction_ss[0], "pred_q01": prediction_ss[1]},
    {"name": "CNN-LTT", "pred_lambda0": prediction_ltt[0], "pred_q01": prediction_ltt[1]},
    {"name": "GNN-PhyloPool", "pred_lambda0": prediction_gnn_PP[0], "pred_q01": prediction_gnn_PP[1]},
    {"name": "GNN-avg", "pred_lambda0": prediction_gnn_ism[0], "pred_q01": prediction_gnn_ism[1]},
    {"name": "CNN-CDV", "pred_lambda0": prediction_cblv[0], "pred_q01": prediction_cblv[1]},
    {"name": "MLE", "pred_lambda0":  pred['lambda0_pred'], "pred_q01":  pred['q01_pred']},
]


true_lambda0 = true[0]
true_q01 = true[1]

# Calculer l'erreur absolue et relative pour chaque modèle
errors_lambda0_abs = []
errors_q01_abs = []
errors_lambda0_rel = []
errors_q01_rel = []
methods = []

for model in prediction_models:
    if model["name"] == "MLE":
        name = model["name"]
        pred_lambda0 = model["pred_lambda0"]
        pred_q01 = model["pred_q01"]

        error_lambda0_abs = np.abs((pred_lambda0 - true_mle["lambda0"]))
        error_q01_abs = np.abs((pred_q01 - true_mle["q01"]))
        error_lambda0_rel = np.abs((pred_lambda0 - true_mle["lambda0"])) / true_mle["lambda0"]
        error_q01_rel = np.abs((pred_q01 - true_mle["q01"])) / true_mle["q01"]

        errors_lambda0_abs.append(error_lambda0_abs)
        errors_q01_abs.append(error_q01_abs)
        errors_lambda0_rel.append(error_lambda0_rel)
        errors_q01_rel.append(error_q01_rel)
        methods.append([name] * len(true_lambda0))

    else:
        name = model["name"]
        pred_lambda0 = model["pred_lambda0"]
        pred_q01 = model["pred_q01"]

        error_lambda0_abs = np.abs((pred_lambda0 - true_lambda0))
        error_q01_abs = np.abs((pred_q01 - true_q01))
        error_lambda0_rel = np.abs((pred_lambda0 - true_lambda0)) / true_lambda0
        error_q01_rel = np.abs((pred_q01 - true_q01)) / true_q01

        errors_lambda0_abs.append(error_lambda0_abs)
        errors_q01_abs.append(error_q01_abs)
        errors_lambda0_rel.append(error_lambda0_rel)
        errors_q01_rel.append(error_q01_rel)
        methods.append([name] * len(true_lambda0))

# Convertir les erreurs et les méthodes en DataFrame
df_lambda0_abs = pd.DataFrame({
    "method": np.concatenate(methods),
    "error_lambda0_abs": np.concatenate(errors_lambda0_abs)
})
df_q01_abs = pd.DataFrame({
    "method": np.concatenate(methods),
    "error_q01_abs": np.concatenate(errors_q01_abs)
})
df_lambda0_rel = pd.DataFrame({
    "method": np.concatenate(methods),
    "error_lambda0_rel": np.concatenate(errors_lambda0_rel)
})
df_q01_rel = pd.DataFrame({
    "method": np.concatenate(methods),
    "error_q01_rel": np.concatenate(errors_q01_rel)
})

# Calculer la MAE et la MRE pour chaque méthode et stocker dans un dictionnaire
mae_lambda0 = {name: np.mean(errors_lambda0_abs[i]) for i, name in enumerate([model['name'] for model in prediction_models])}
mae_q01 = {name: np.mean(errors_q01_abs[i]) for i, name in enumerate([model['name'] for model in prediction_models])}
mre_lambda0 = {name: np.mean(errors_lambda0_rel[i]) for i, name in enumerate([model['name'] for model in prediction_models])}
mre_q01 = {name: np.mean(errors_q01_rel[i]) for i, name in enumerate([model['name'] for model in prediction_models])}
plt.figure(figsize=(20, 10))

colors = ['skyblue', 'coral', 'lightgreen', 'gold', 'lightcoral', 'cyan']
model_order = ["MLP-SS", "CNN-LTT", "CNN-CBLV", "GNN-avg", "GNN-PhyloPool", "MLE"]

plt.rcParams.update({
    'font.size': 16,  # Augmente la taille de tous les textes
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

plt.figure(figsize=(20, 10))


colors = ['skyblue', 'coral', 'lightgreen', 'gold', 'lightcoral', 'cyan']
model_order = ["MLP-SS", "CNN-LTT", "CNN-CDV", "GNN-avg", "GNN-PhyloPool", "MLE"]

# Erreur absolue pour Lambda0
plt.subplot(2, 2, 1)
sns.violinplot(data=df_lambda0_abs, x='method', y='error_lambda0_abs', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mae = np.floor(mae_lambda0[name] * 1000) / 1000
    plt.text(i, mae_lambda0[name], f"$\\bf{{{truncated_mae:.3f}}}$", ha='center', fontsize=16, color="black") 
plt.title("Distribution of Absolute Error for Lambda0")
plt.xlabel("")
plt.ylabel("Absolute Error")

# Erreur relative pour Lambda0
plt.subplot(2, 2, 2)
sns.violinplot(data=df_lambda0_rel, x='method', y='error_lambda0_rel', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mre = np.floor(mre_lambda0[name] * 1000) / 1000
    plt.text(i, mre_lambda0[name], f"$\\bf{{{truncated_mre:.3f}}}$", ha='center', fontsize=16, color="black")  
plt.title("Distribution of Relative Error for Lambda0")
plt.xlabel("")
plt.ylabel("Relative Error")

# Erreur absolue pour q01
plt.subplot(2, 2, 3)
sns.violinplot(data=df_q01_abs, x='method', y='error_q01_abs', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mae = np.floor(mae_q01[name] * 10000) / 10000
    plt.text(i, mae_q01[name], f"$\\bf{{{truncated_mae:.4f}}}$", ha='center', fontsize=16, color="black") 
plt.title("Distribution of Absolute Error for q01")
plt.xlabel("")
plt.ylabel("Absolute Error")

# Erreur relative pour q01
plt.subplot(2, 2, 4)
sns.violinplot(data=df_q01_rel, x='method', y='error_q01_rel', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mre = np.floor(mre_q01[name] * 100) / 100
    plt.text(i, mre_q01[name], f"$\\bf{{{truncated_mre:.2f}}}$", ha='center', fontsize=16, color="black") 
plt.title("Distribution of Relative Error for q01")
plt.xlabel("")
plt.ylabel("Relative Error")

plt.tight_layout()
plt.show()
