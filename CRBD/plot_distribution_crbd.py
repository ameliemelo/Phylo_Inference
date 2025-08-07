import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Charger les données
true = np.load("/CRBD/results/true_crbd.npy")
true_lambda = np.array(true[0])
true_mu = np.array(true[1])

# Préparer les prédictions
models = [
    {"name": "MLP_SS", "pred": np.load("/CRBD/results/pred_crbd_MLP_ss.npy")},
    {"name": "CNN_LTT", "pred": np.load("/CRBD/results/pred_crbd_CNN_ltt.npy")[::-1]},  # inversé
    {"name": "GNN_PhyloPool", "pred": np.load("/CRBD/results/pred_crbd_GNN_PhyloPool.npy")[::-1]},
    {"name": "GNN_AVG", "pred": np.load("/CRBD/results/pred_crbd_GNN_avg.npy")[::-1]},
    {"name": "CNN_CBLV", "pred": np.load("/CRBD/results/pred_crbd_CNN_cblv.npy")},
]

# Charger les prédictions MLE
with open('/CRBD/results/dataMLE_crbd.json', 'r') as file:
    data = json.load(file)
models.append({"name": "MLE", "pred": np.array(data['pred'])})

# Plot predicted lambda vs true lambda
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, model in enumerate(models):
    pred_lambda = np.array(model["pred"][0])

    axs[i].scatter(true_lambda, pred_lambda, s=5, alpha=0.3, color="blue")
    axs[i].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)  # y = x line
    axs[i].set_xlabel("True value of lambda")
    axs[i].set_ylabel("Predicted value of lambda")
    axs[i].set_title(f"Lambda prediction: {model['name']}")
    # axs[i].set_xlim(-0.1, 1.2)
    # axs[i].set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.show()

# Plot predicted mu vs true mu
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, model in enumerate(models):
    pred_mu = np.array(model["pred"][1])

    axs[i].scatter(true_mu, pred_mu, s=5, alpha=0.3, color="green")
    axs[i].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)  # y = x line
    axs[i].set_xlabel("True value of mu")
    axs[i].set_ylabel("Predicted value of mu")
    axs[i].set_title(f"Mu prediction: {model['name']}")

plt.tight_layout()
plt.show()


# Code to see the distribution of errors depending on the true values

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# axs = axs.flatten()
# for i, model in enumerate(models):
#     pred_lambda = np.array(model["pred"][0])
#     errors_lambda = np.abs(true_lambda - pred_lambda)

#     axs[i].scatter(true_lambda, errors_lambda, s=5, alpha=0.3, color="blue")
#     axs[i].set_xlabel("Valeur vraie de lambda")
#     axs[i].set_ylabel("Erreur absolue")
#     axs[i].set_title(f"Erreur lambda : {model['name']}")
#     axs[i].set_xlim(0, 1)
#     axs[i].set_ylim(0, 0.35)

# plt.tight_layout()
# plt.show()


# # Tracer la densité KDE pour l'erreur lambda
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# axs = axs.flatten()

# for i, model in enumerate(models):
#     pred_mu = np.array(model["pred"][1])
#     errors_mu = np.abs(true_mu - pred_mu)

#     axs[i].scatter(true_mu, errors_mu, s=5, alpha=0.3, color="blue")
#     axs[i].set_xlabel("Valeur vraie de mu")
#     axs[i].set_ylabel("Erreur absolue")
#     axs[i].set_title(f"Erreur mu : {model['name']}")
#     axs[i].set_xlim(0, 1)
#     axs[i].set_ylim(0, 0.35)

# plt.tight_layout()
# plt.show()
