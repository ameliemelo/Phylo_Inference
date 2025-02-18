
source("R/phylo-inference-ml.R")
set.seed(1)


phylo <- readRDS("data/MLE-100k-bisse.rds") # change file name if needed
true_params <- readRDS("data/true-parameters-100k-crbd.rds") # same
cat("Taille de la phylo: ", length(phylo), "\n")

# Summary statistics
sumstat <- generateSumStatFromPhylo(phylo, true_params) 
saveRDS(sumstat, "data/new-phylogeny-100tree-sumstat.rds")

# CBLV
max_taxa <- 1000 # maximum phylogeny size, change if needed
# If model == "crbd", do call 
cblv <- generate_encoding(phylo, max_taxa) 

# If model == "bisse", do call 
cblv <- generate_encoding_bisse(phylo, max_taxa)
saveRDS(cblv, "data/cblv-conca.rds")



# LTT
taxa_range <- c(10,20) # range of phylogeny size
df.ltt <- generate_ltt_dataframe(phylo, taxa_range, true_params)$ltt
saveRDS(df.ltt, "data/bisse_ltt.rds")


# Graph

# If model == "bisse", add a line in convert-phylo-to-graph.R within the function get_node_df  
graphs <- generate_phylogeny_graph(phylo)
saveRDS(graphs, "data/test10tree_bisse_graph.rds")





