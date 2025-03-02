library(jsonlite)
library(reticulate)
numpy <- import("numpy")
source("R/phylo-inference-ml.R")
set.seed(113)

phylo <- readRDS("/home/amelie/These/Phylo_Inference/data/tree-100k-bisse.rds") 
true_params <- readRDS("/home/amelie/These/Phylo_Inference/data/true-parameters-100k-bisse.rds")
print(length(phylo))

# Charger le fichier numpy .npy
test_ind <- numpy$load("/home/amelie/These/Phylo_Inference/BiSSE/test_indices.npy")
print(test_ind)

trees <- list()  # Créer une liste pour stocker les arbres

for (i in test_ind) {
  trees <- append(trees, list(phylo[[i]]))
}


# Faire une estimation MLE uniquement sur les arbres avec les indices test_ind
pred_params <- getPredsMLE(type = "bisse", trees)

lambda0_pred <- pred_params[[1]]
q01_pred <- pred_params[[5]]

# Extraction des vraies valeurs
lambda0_true <- true_params$lambda0[test_ind]
q01_true <- true_params$q01[test_ind]

# Tracer les graphiques
par(mfrow=c(1,2)) 

plot(lambda0_true, lambda0_pred, main="Lambda - True vs Predicted", 
     xlab="True Lambda0", ylab="Predicted Lambda0", col="blue")
abline(0, 1, col="red") 

plot(q01_true, q01_pred, main="q01 - True vs Predicted",
     xlab="True q01", ylab="Predicted q01", col="blue")
abline(0, 1, col="red")


pred <- list(lambda0_pred,q01_pred)
true <- list(lambda0 = lambda0_true, q01 = q01_true)

# Créer une liste contenant pred et true
data <- list(pred = pred, true = true)

# Enregistrer les données au format JSON
write_json(data, "dataMLE_bisse.json")