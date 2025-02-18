
# Import libraries
source("R/phylo-inference-ml.R")

# Set up
model <- "bisse" # type of the model, either: "crbd" or "bisse"
n_trees <-10 # number of trees to generate
n_taxa <- c(100, 200) # range size of the generated phylogenies 100 a 1000

# Define space parameters.
# For the CRBD model
lambda_range <- c(0.1, 1.0) # speciation rate
epsilon_range <- c(0.0, 0.9) # turnover rate
param.range.crbd <- list(
  "lambda" = lambda_range,
  "epsilon" = epsilon_range
)

# For the BiSSE model
lambda_range <- c(0.1, 1.) # speciation rate
q_range <- c(0.01, 0.1) # transition rate
param.range.bisse <- list(
  "lambda" = lambda_range,
  "q" = q_range
)

# Select the parameter space of the chosen diversification model
param.range.list <- list(
  "crbd" = param.range.crbd,
  "bisse" = param.range.bisse
)
param.range <- param.range.list[[model]]

# Generating and saving phylogenies
out <- generatePhylo(model, n_trees, n_taxa, param.range)
phylo <- out$trees
params <- out$param


saveRDS(phylo, paste("data/test10tree-", model, "-new.rds", sep=""))
saveRDS(params, paste("data/test10tree-true-parameters--", model, "-new.rds", sep=""))

