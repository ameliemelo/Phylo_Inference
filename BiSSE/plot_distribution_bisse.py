import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Charger les données
true = np.load("/BiSSE/results/true_bisse.npy")
true_lambda1 = np.array(true[0])
true_q011 = np.array(true[1])

# Préparer les prédictions
models = [
    {"name": "MLP_SS", "pred": np.load("/BiSSE/results/pred_bisse_MLP_ss.npy")},
    {"name": "CNN_LTT", "pred": np.load("/BiSSE/results/pred_bisse_CNN_ltt.npy")},  
    {"name": "GNN_PhyloPool", "pred": np.load("/BiSSE/results/pred_bisse_GNN_PhyloPool2.npy")},
    {"name": "GNN_AVG", "pred": np.load("/BiSSE/results/pred_bisse_GNN_avg.npy")},
    {"name": "CNN_CBLV", "pred": np.load("/BiSSE/results/pred_bisse_CNN_cblv.npy")},
]

with open('/BiSSE/results/dataMLE_bisse.json', 'r') as file:
    data = json.load(file)

pred = data['pred']
true_mle = {
    'lambda0': np.array(data['true']['lambda0']),
    'q01': np.array(data['true']['q01'])
}

pred_lambda0 = np.array(pred['lambda0_pred'])
pred_q01 = np.array(pred['q01_pred'])

# Ajouter le modèle MLE au tableau des modèles
models.append({
    "name": "MLE",
    "pred": [pred_lambda0, pred_q01]
})
# Plot predicted lambda vs true lambda
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, model in enumerate(models):
    if model["name"] == "MLE":
        name = model["name"]
        pred_lambda = model["pred"][0]
        true_lambda = true_mle["lambda0"]
    else:
        pred_lambda = np.array(model["pred"][0])
        true_lambda = true_lambda1

    axs[i].scatter(true_lambda, pred_lambda, s=5, alpha=0.3, color="blue")
    axs[i].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)  # y = x line
    axs[i].set_xlabel("True value of lambda0")
    axs[i].set_ylabel("Predicted value of lambda0")
    axs[i].set_title(f"Lambda0 prediction: {model['name']}")


plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, model in enumerate(models):
    if model["name"] == "MLE":
        name = model["name"]
        pred_q01 = model["pred"][1]
        true_q01 = true_mle["q01"]
    else:
        pred_q01 = np.array(model["pred"][1])
        true_q01 = true_q011

    min_x, max_x = true_q01.min(), true_q01.max()
    min_y, max_y = pred_q01.min(), pred_q01.max()

    # Ajouter 10% de marge
    dx = (max_x - min_x) * 0.1 if max_x > min_x else 1e-3
    dy = (max_y - min_y) * 0.1 if max_y > min_y else 1e-3

    axs[i].scatter(true_q01, pred_q01, s=5, alpha=0.3, color="green")

    lower = min(min_x, min_y)
    upper = max(max_x, max_y)
    axs[i].plot([lower, upper], [lower, upper], color='red', linestyle='--', linewidth=1)

    axs[i].set_xlim(min_x - dx, max_x + dx)
    axs[i].set_ylim(min_y - dy, max_y + dy)

    axs[i].set_xlabel("True value of q01")
    axs[i].set_ylabel("Predicted value of q01")
    axs[i].set_title(f"q01 prediction: {model['name']}")

plt.tight_layout()
plt.show()
