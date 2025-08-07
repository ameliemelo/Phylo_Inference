import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import seaborn as sns


true= np.load("/CRBD/results/true_crbd.npy")
print(true)
print(len(true[0]))


pred_ss = np.load("/CRBD/results/pred_crbd_MLP_ss.npy")

prediction_ss =[[],[]]
prediction_ss[1] = pred_ss[1]
prediction_ss[0] = pred_ss[0]

pred_ltt = np.load("/CRBD/results/pred_crbd_CNN_ltt.npy")

prediction_ltt  =[[],[]]
prediction_ltt[1] = pred_ltt[0]
prediction_ltt[0] = pred_ltt[1]

pred_gnn_PP = np.load("/CRBD/results/pred_crbd_GNN_PhyloPool.npy")


prediction_gnn_PP =[[],[]]
prediction_gnn_PP[1] = pred_gnn_PP[0]
prediction_gnn_PP[0] = pred_gnn_PP[1]

pred_gnn_ism = np.load("/CRBD/results/pred_crbd_GNN_avg.npy")


prediction_gnn_ism =[[],[]]
prediction_gnn_ism[1] = pred_gnn_ism[0]
prediction_gnn_ism[0] = pred_gnn_ism[1]

pred_cblv = np.load("/CRBD/results/pred_crbd_CNN_cblv.npy") #lr = 0.0005

prediction_cblv  =[[],[]]
prediction_cblv[1] = pred_cblv[1]
prediction_cblv[0] = pred_cblv[0]

import numpy as np
import json
import matplotlib.pyplot as plt
with open('/CRBD/results/dataMLE_crbd.json', 'r') as file:
    data = json.load(file)

pred = data['pred']
print(len(pred[0]))


# Exemple de structure de données
prediction_models = [
    {"name": "MLP-SS", "pred_lambda": prediction_ss[0], "pred_mu": prediction_ss[1]},
    {"name": "CNN-LTT", "pred_lambda": prediction_ltt[0], "pred_mu": prediction_ltt[1]},
    {"name": "GNN-PhyloPool", "pred_lambda": prediction_gnn_PP[0], "pred_mu": prediction_gnn_PP[1]},
    {"name": "GNN-avg", "pred_lambda": prediction_gnn_ism[0], "pred_mu": prediction_gnn_ism[1]},
    {"name": "CNN-CDV", "pred_lambda": prediction_cblv[0], "pred_mu": prediction_cblv[1]},
    {"name": "MLE", "pred_lambda":  np.array(pred[0]), "pred_mu":  np.array(pred[1])},
]


true_lambda = true[0]
true_mu = true[1]

colors = ['skyblue', 'coral', 'lightgreen', 'gold', 'lightcoral', 'cyan']
model_order = ["MLP-SS", "CNN-LTT", "CNN-CDV", "GNN-avg", "GNN-PhyloPool", "MLE"]
MAX_ERROR = 4  # Seuil maximal d'affichage des erreurs pour Mu

# Calculer les erreurs absolues et relatives pour chaque modèle en utilisant des dictionnaires
errors_lambda_abs = {}
errors_mu_abs = {}
errors_lambda_rel = {}
errors_mu_rel = {}

for model in prediction_models:
    name = model["name"]
    error_lambda_abs = np.abs(model["pred_lambda"] - true_lambda)
    error_mu_abs = np.abs(model["pred_mu"] - true_mu)
    error_lambda_rel = error_lambda_abs / true_lambda
    error_mu_rel = error_mu_abs / true_mu

    errors_lambda_abs[name] = error_lambda_abs
    errors_mu_abs[name] = error_mu_abs
    errors_lambda_rel[name] = error_lambda_rel
    errors_mu_rel[name] = error_mu_rel

# Créer les DataFrames à partir des dictionnaires
df_lambda_abs = pd.DataFrame({
    "method": np.repeat(list(errors_lambda_abs.keys()), len(true_lambda)),
    "error_lambda_abs": np.concatenate(list(errors_lambda_abs.values()))
})
df_mu_abs = pd.DataFrame({
    "method": np.repeat(list(errors_mu_abs.keys()), len(true_mu)),
    "error_mu_abs": np.concatenate(list(errors_mu_abs.values()))
})
df_lambda_rel = pd.DataFrame({
    "method": np.repeat(list(errors_lambda_rel.keys()), len(true_lambda)),
    "error_lambda_rel": np.concatenate(list(errors_lambda_rel.values()))
})
df_mu_rel = pd.DataFrame({
    "method": np.repeat(list(errors_mu_rel.keys()), len(true_mu)),
    "error_mu_rel": np.concatenate(list(errors_mu_rel.values()))
})

# Calculer la MAE et la MRE pour chaque méthode et stocker dans un dictionnaire
mae_lambda = {name: np.mean(errors_lambda_abs[name]) for name in model_order}
mae_mu = {name: np.mean(errors_mu_abs[name]) for name in model_order}
mre_lambda = {name: np.mean(errors_lambda_rel[name]) for name in model_order}
mre_mu = {name: np.mean(errors_mu_rel[name]) for name in model_order}

plt.rcParams.update({
    'font.size': 16,  
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})
# Affichage des graphiques
plt.figure(figsize=(20, 10))  

# Erreur absolue pour Lambda
plt.subplot(2, 2, 1)
sns.violinplot(data=df_lambda_abs, x='method', y='error_lambda_abs', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mae = np.floor(mae_lambda[name] * 1000) / 1000  
    plt.text(i, mae_lambda[name], f"$\\bf{{{truncated_mae:.3f}}}$", ha='center', fontsize=16, color="black")  
plt.title("Distribution of Absolute Error for Lambda")
plt.xlabel("") 
plt.ylabel("Absolute Error")

# Erreur relative pour Lambda
plt.subplot(2, 2, 2)
sns.violinplot(data=df_lambda_rel, x='method', y='error_lambda_rel', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mre = np.floor(mre_lambda[name] * 1000) / 1000  
    plt.text(i, mre_lambda[name], f"$\\bf{{{truncated_mre:.3f}}}$", ha='center', fontsize=16, color="black") 
plt.title("Distribution of Relative Error for Lambda")
plt.xlabel("") 
plt.ylabel("Relative Error")

# Erreur absolue pour Mu
plt.subplot(2, 2, 3)
sns.violinplot(data=df_mu_abs, x='method', y='error_mu_abs', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mae = np.floor(mae_mu[name] * 1000) / 1000  
    plt.text(i, mae_mu[name], f"$\\bf{{{truncated_mae:.3f}}}$", ha='center', fontsize=16, color="black") 
plt.title("Distribution of Absolute Error for Mu")
plt.xlabel("")  
plt.ylabel("Absolute Error")

# Erreur relative pour Mu
plt.subplot(2, 2, 4)
sns.violinplot(data=df_mu_rel[df_mu_rel['error_mu_rel'] <= MAX_ERROR], x='method', y='error_mu_rel', order=model_order, palette=colors, inner=None, cut=0)
for i, name in enumerate(model_order):
    truncated_mre = np.floor(mre_mu[name] * 100) / 100 
    plt.text(i, mre_mu[name], f"$\\bf{{{truncated_mre:.2f}}}$", ha='center', fontsize=16, color="black")  
    # Afficher la valeur max supprimée au-dessus de MAX_ERROR
    max_removed = df_mu_rel[(df_mu_rel['method'] == name) & (df_mu_rel['error_mu_rel'] > MAX_ERROR)]['error_mu_rel'].max()
    if not pd.isna(max_removed):
        plt.text(i, MAX_ERROR + 0.2, f"Max: {max_removed:.2f}", ha='center', fontsize=12, color="red")
plt.title("Distribution of Relative Error for Mu")
plt.xlabel("")  
plt.ylabel("Relative Error")
plt.ylim(0, MAX_ERROR + 1)

plt.tight_layout()
plt.show()



