library(jsonlite)
library(reticulate)
numpy <- import("numpy")
source("R/phylo-inference-ml.R")
set.seed(113)

phylo <- readRDS("data/tree-100k-crbd.rds") 
true_params <- readRDS("data/true-parameters-100k-crbd.rds")
print(length(phylo))

# Charger le fichier numpy .npy
test_ind <- numpy$load("/home/amelie/These/Phylo_Inference/CRBD/test_indices.npy")
print(test_ind)

trees <- list()  # Créer une liste pour stocker les arbres

for (i in test_ind) {
  trees <- append(trees, list(phylo[[i]]))
}


# Faire une estimation MLE uniquement sur les arbres avec les indices test_ind
pred_params <- getPredsMLE(type = "crbd", trees)

lambda_pred <- pred_params[[1]]
mu_pred <- pred_params[[2]]

# Extraction des vraies valeurs
lambda_true <- true_params$lambda[test_ind]
mu_true <- true_params$mu[test_ind]

# Tracer les graphiques
par(mfrow=c(1,2)) 

plot(lambda_true, lambda_pred, main="Lambda - True vs Predicted", 
     xlab="True Lambda", ylab="Predicted Lambda", col="blue")
abline(0, 1, col="red") 

plot(mu_true, mu_pred, main="Mu - True vs Predicted",
     xlab="True Mu", ylab="Predicted Mu", col="blue")
abline(0, 1, col="red")


pred <- list(lambda_pred,mu_pred)
true <- list(lambda = lambda_true, mu = mu_true)

# Créer une liste contenant pred et true
data <- list(pred = pred, true = true)

# Enregistrer les données au format JSON
write_json(data, "dataMLE.json")