#' Create the edges dataframe of a phylo tree 
#'
#' For a graph with N edges, the dataframe has a shape (N, 2). One row 
#' corresponds to one edge, the two columns give the indexes of the two nodes 
#' linked. 
#'
#' @param tree phylo tree 
#'              
#' @return df (edges data frame)
#' 
#' @export
#' @examples
get_edge_df <- function(tree){
  df <- as.data.frame(tree$edge)
  colnames(df) <- c("node1", "node2")
  return(df)
}


#' Create the nodes dataframe of a phylo tree 
#'
#' For a graph with N and considering M attributes per node, the dataframe has a
#' shape (N, M). One row corresponds to one node, each column corresponds to one 
#' attribute. 
#' Examples of nodes attribute: distance to root, number of children, etc.
#'
#' @param tree phylo tree 
#'              
#' @return df (nodes data frame)
#' 
#' @export
#' @examples
get_node_df <- function(tree){
  n_nodes <- 2*tree$Nnode+1
  name.attr <- c("dist", "mean.edge", "time.asym",
                 "clade.asym", "ancestor", "descendant","state")
  n_attr    <- length(name.attr)
  nodes.attr <- vector(mode = "list", length = n_attr)
  names(nodes.attr) <- name.attr
  nodes.attr$dist <- get_all_distances_to_root(tree, as_edge_count = FALSE)
  nodes.attr$ancestor <- get_all_distances_to_root(tree, as_edge_count = TRUE)
  for (i in 1:n_nodes){
    descendant <- length(Descendants(tree, i, "all"))
    #in.edge <- get_in.edge.length(tree, i)
    mean.edge <- get_mean.edge.length(tree, i)
    time.asym <- get_time.asym(tree, i)
    clade.asym <- get_clade.asym(tree, i)
    state <- get_node_state(tree, i)
    nodes.attr$descendant <- c(nodes.attr$descendant, descendant)
    nodes.attr$clade.asym <- c(nodes.attr$clade.asym, clade.asym)
    nodes.attr$time.asym  <- c(nodes.attr$time.asym, time.asym)
    nodes.attr$mean.edge  <- c(nodes.attr$mean.edge, mean.edge)
    #nodes.attr$in.edge    <- c(nodes.attr$in.edge, in.edge)

    # Add the following line if the BiSSE model is used  
    # nodes.attr$state <- c(nodes.attr$state, state)

  }
  df <- as.data.frame(nodes.attr)
  return(df)
}

is_root <- function(tree, node){
  return(node == Ntip(tree) + 1)
}

get_in.edge.length <- function(tree, node){
  if (is_root(tree, node)){length <- 0.} # no incoming edge if node is root 
  else{
    parent <- Ancestors(tree, node, "parent") # get node's parent 
    bool   <- tree$edge[,1] == parent & tree$edge[,2] == node
    index  <- which(bool == TRUE) # edge index 
    length <- tree$edge.length[index] # edge length 
  }
  return(length)
}

get_mean.edge.length <- function(tree, node){
  
  # Node is root - mean of two outcoming edges 
  if (is_root(tree, node)){
    children <- Children(tree, node) # get children
    length1  <- get_in.edge.length(tree, children[1]) # length: root --> child1
    length2  <- get_in.edge.length(tree, children[2]) # length: root --> child2
    mean     <- mean(c(length1, length2)) # compute the mean 
  }
  
  # Node is tip - unique incoming edge 
  else if (is_tip(node, tree)){
    mean <- get_in.edge.length(tree, node)
  }
  
  # Node is internal node - mean of two outcoming edges and one incoming edge 
  else {
    children <- Children(tree, node) # get children
    length1  <- get_in.edge.length(tree, children[1]) # length: node --> child1
    length2  <- get_in.edge.length(tree, children[2]) # length: node --> child2
    length3  <- get_in.edge.length(tree, node)        # length: parent --> node
    mean     <- mean(c(length1, length2, length3)) # compute the mean 
  }
  return(mean)
}


get_clade.asym <- function(tree, node){
  
  # If node has no child, returns 0
  if (is_tip(node, tree)){
    asym <- 0.
  }
  
  # Else compute the branch asymmetry 
  else{
    children <- Children(tree, node) # get children
    length1  <- get_in.edge.length(tree, children[1]) # length: node --> child1
    length2  <- get_in.edge.length(tree, children[2]) # length: node --> child2
    asym     <- abs(length1 - length2) / mean(c(length1, length2))
  }
  return(asym)
}


get_time.asym <- function(tree, node){
  if (is_tip(node, tree) | is_root(tree, node)){
    asym <- 0.
  }
  else{ 
    length1 <- get_in.edge.length(tree, node) # length: parent --> node
    children <- Children(tree, node) # get children
    length2  <- get_in.edge.length(tree, children[1]) # length: node --> child1
    length3  <- get_in.edge.length(tree, children[2]) # length: node --> child2
    asym     <- abs(length1 - mean(c(length2, length3))) / mean(c(length1, length2, length3))
  }
  return(asym)
}

get_node_state <- function(tree, i){
  if (is_tip(i, tree)){state <- tree$tip.state[[i]]} # node is tip, save state
  else{state <- -1} # node isn't tip, state is unknown (i.e. -1)
  return(state)
}

generate_phylogeny_graph <- function(phylogenies) {
  out <- list()
  n <- length(phylogenies)
  for (i in 1:n){
    progress(i, max.value = n, progress.bar = TRUE, init = (i == 1))
    phylo <- phylogenies[[i]]
    df.edge <- get_edge_df(phylo)
    df.node <- get_node_df(phylo)
    df.list <- list("node" = df.node, "edge" = df.edge)
    out[[i]] <- df.list
  }
  out
}

# 
# # Simulations parameters 
# n_trees  <- 10009 # number of trees 
# n_taxa   <- c(100,1000) # taxa range
# param.range <- list("lambda" = c(0.1,1.),
#                     "epsilon"= c(0.,.9)) # range of each parameters 
# ss_check <- TRUE # no NAs in the summary statistics?
# 
# # Preparation 
# out <- readPhylogeny(n_trees, n_taxa, param.range) # load trees 
# trees <- out$trees # extract 
# list.df.tree <- list() # where dataframe will be stored 
# 
# 
# # Computing the node and edge data frame for each tree and save them to the list
# for (i in 1:n_trees){
#   progress(i, max.value = n_trees, progress.bar = TRUE, init = (i == 1))
#   tree <- trees[[i]]
#   df.edge <- get_edge_df(tree)
#   df.node <- get_node_df(tree)
#   df.list <- list("node" = df.node, "edge" = df.edge)
#   list.df.tree[[i]] <- df.list
# }
# 
# # Saving 
# dir   <- "trees-dataset"
# fname <- get_backbone_save_name(n_trees, n_taxa, param.range) # save name for file 
# fname <- paste(fname, "sscheck", ss_check, "df.rds", sep = "-")
# fname <- paste(dir, fname, sep = "/")
# saveRDS(list.df.tree, fname) # saving file 
# cat(paste(fname, " saved.\n"))
