
source("R/phylo-inference-ml.R")
set.seed(1)


phylo <- readRDS("data/tree-100k-crbd.rds") # change file name if needed
true_params <- readRDS("data/true-parameters-100k-crbd.rds") # same
max_taxa <- 1000 # maximum phylogeny size, change if needed


# Summary statistics
sumstat <- generateSumStatFromPhylo(phylo, true_params) 

# If model == "crbd", do call 
sumstat <- scale_summary_statistics(sumstat, max_taxa, c("lambda", "mu"))

# # If model == "bisse", do call
# sumstat <- df_add_tipstate(sumstat, phylo)
# sumstat <- scale_summary_statistics(sumstat, max_taxa, c("lambda0", "q01"))

saveRDS(sumstat, "data/sumstat-100k-crbd.rds")

# CBLV
# If model == "crbd", do call 
cblv <- generate_encoding(phylo, max_taxa) 

# # If model == "bisse", do call 
# cblv <- generate_encoding_bisse(phylo, max_taxa)
saveRDS(cblv, "data/cblv-100k-crbd.rds")



# LTT
taxa_range <- c(100, 1000) # range of phylogeny size
df.ltt <- generate_ltt_dataframe(phylo, taxa_range, true_params)$ltt
saveRDS(df.ltt, "data/ltt-100k-crbd.rds")



# Graph

# If model == "bisse", add a line in convert-phylo-to-graph.R within the function get_node_df  and change name.attr
graphs <- generate_phylogeny_graph(phylo)
saveRDS(graphs, "data/graph-100k-crbd.rds")




