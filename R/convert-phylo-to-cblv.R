# This file contains the function to encode a phylogenetic tree with the
# "Compact Bijective Ladderized Vector" methods
# See Voznica 2021 - bioRxiv - DOI:10.1101/2021.03.11.435006 



#### Encoding Tree ####

#' Check a node of a tree is a tip
#'
#'
#' @param node integer, index of the node
#' @param tree phylo tree
#'
#' @return logical 
#' @export
#' @examples
is_tip <- function(node, tree){
  node <= length(tree$tip.label)
}

#' Get the two children of a node
#'
#' The left children is always the node the further from the root
#' The right children is always the node the closest to the root
#' If the node is a tip returns NA values 
#'
#' @param node integer, index of the node
#' @param tree phylo tree
#'
#' @return list, $left = left child, $right = right child
#' @export
#' @examples
get_child <- function(node, tree){
  child <- list("left" = NA, "right" = NA)
  dist.all <- castor::get_all_distances_to_root(tree)
  
  # First check that the node is not a tip (otherwise returns NA)
  if (!is_tip(node, tree)){ 
    idx <- phangorn::Children(tree, node)
    
    # If both child are tips (same dist. to root), random attribution (l/r)
    if (is_tip(idx[1], tree) & is_tip(idx[2], tree)){
      child$left  <- idx[1]
      child$right <- idx[2]
    }
    
    # Else (>= 1 child is internal), left child is the further from root
    else {
      dist.child <- c(dist.all[idx[1]], dist.all[idx[2]])
      left  <- which(dist.child == max(dist.child))
      right <- which(dist.child == min(dist.child))
      child$left  <- idx[left]
      child$right <- idx[right]
    }
  }
  return(child) 
}


#' Traverse a tree inorder 
#' 
#'
#' @param tree phylo tree
#'
#' @return vector, ordered sequence of node indexes of the inorder traversal^
#' @export
#' @examples
traverse_inorder <- function(tree){
  node  <- length(tree$tip.label) + 1 # root index 
  stack <- c()
  inorder <- c()
  while(length(stack) != 0 | !is.na(node)){
    if(!is.na(node)){
      stack <- c(node, stack)
      node <- get_child(node, tree)$left
    }
    else{
      node <- stack[1]
      inorder <- c(inorder, node)
      stack <- stack[stack!=node]
      node <- get_child(node, tree)$right
    }
  }
  return(inorder)
}


#' Compute all the distances of tips to their most recent ancestor 
#'
#' For each node, 
#' 1. find its parent (most recent ancestor)
#' 2. compute the distance between the node and its parent 
#' This distance is only computed for tips, as we don't need this distance 
#' for the internal nodes in the encoding. 
#' Thus internal node indexes are filled with NA 
#' See Voznica 2021 - bioRxiv - DOI:10.1101/2021.03.11.435006 
#'
#' @param tree phylo tree
#'
#' @return vector of distance 
#' @export
#' @examples
get_all_distances_to_ancestor <- function(tree){
  n_node <- tree$Nnode # number of internal nodes
  n_tip  <- n_node + 1 # number of tips (size of the tree)
  n_tot  <- n_tip + n_node # total number of nodes 
  dist <- c()
  
  # Fill tips value w/ their distance to their parent node
  for (tip in 1:n_tip){
    parent <- Ancestors(tree, tip, "parent") # get the parent of the tip
    bool   <- tree$edge[,1]==parent & tree$edge[,2]==tip # edge: parent ---> tip
    edge.idx <- which(bool==TRUE) # edge index 
    edge.length <- tree$edge.length[edge.idx] # edge length 
    dist <- c(dist, edge.length) # save
  }
  
  # Fill nodes values w/ NA
  for (i in 1:n_node){ 
    dist <- c(dist, NA)
  }
  return(dist)
}

#' Encode a phylogenetic tree into a vector 
#'
#' Compute the compact encoding of a phylogenetic tree. This encoding is
#' bijective. The encoding methods is named: "Compact Bijective Ladderized 
#' Vector" (CBLV).
#' See Voznica 2021 - bioRxiv - DOI:10.1101/2021.03.11.435006
#'
#' @param tree phylo tree to encode 
#'
#' @return vector containing the encoding
#' @export
#' @examples
encode_phylo <- function(tree){
  # 
  # A list for nodes containing their distances to the root 
  # A list for tips containing their distances to the most recent ancestor 
  inorder <- traverse_inorder(tree)
  tips  <- c()
  nodes <- c()
  dist_to_root <- castor::get_all_distances_to_root(tree)
  dist_to_ancestor <- get_all_distances_to_ancestor(tree)
  for (node in inorder){
    if (is_tip(node, tree)){
      tips <- c(tips, dist_to_ancestor[node])
    }
    else{
      nodes <- c(nodes, dist_to_root[node])
    }
  }
  encoding <- list("nodes" = nodes, "tips" = tips)
  return(encoding)
}

encode_bisse <- function(tree){
  inorder <- traverse_inorder(tree)
  tips   <- c()
  nodes  <- c()
  states <- c()
  dist_to_root <- castor::get_all_distances_to_root(tree)
  dist_to_ancestor <- get_all_distances_to_ancestor(tree)
  for (node in inorder){
    if (is_tip(node, tree)){
      tips <- c(tips, dist_to_ancestor[node])
      if (tree$tip.state[[node]] == 0) { # tip states can be 0 or 1, but we replace them with -1 or 1 to assign 0 to internal nodes
        states <- c(states, -1)
      } else {
        states <- c(states, tree$tip.state[[node]])
      }
    }
    else{
      nodes <- c(nodes, dist_to_root[node])
    }
  }
  encoding <- list("nodes" = nodes, "tips" = tips, "states" = states)
  return(encoding)
}


#' Format the encoding of a phylo tree
#'
#' Convert the encoding to a vector of dimension 1.
#' The length of the vector is equal to the 2*N - 1 
#' where N is the maximum possible number taxa
#'
#' @param tree.encode encoding of a phylo, generated by encode_phylo
#' @param max_taxa maximum possible number of taxa, determines the length 
#'                 of the returned torch tensor 
#'
#' @return encode.vec, vector containing the encoding of the phylo tree
#' @export
#' @examples
format_encode <- function(tree.encode, max_taxa){
  n <- length(tree.encode) # n=2 if CRBD and n=3 if BiSSE
  encode.vec <- rep(0, n*max_taxa)
  for (i in 1:n){
    encode.sublist <- tree.encode[[i]]
    for (j in 1:length(encode.sublist)){
      encode.vec[j + max_taxa*(i-1)] <- encode.sublist[j]
    }
  }
  return(encode.vec)
}


generate_encoding <- function(trees, n_taxa){
  max_taxa <- max(n_taxa)
  n_trees <- length(trees)
  list.encode <- list()

  cat("Computing encoding vectors...\n")
  
  for (n in 1:n_trees){
    progress(n, n_trees, progress.bar = TRUE, init = (n==1))
    tree <- trees[[n]] # extract tree
    tree.encode   <- encode_phylo(tree) # encode the tree
    format.encode <- format_encode(tree.encode, max_taxa) # format the encoding
    list.encode[[n]] <- format.encode # save to list 
  }
  
  # Convert the list of vectors to a torch tensor 
  tensor.encode <- as.data.frame(do.call(cbind, list.encode)) %>% 
    as.matrix() 
  
  cat("\nComputing encoding vectors... Done.\n")
  
  return(tensor.encode)
}


generate_encoding_bisse <- function(trees, n_taxa){
  max_taxa <- max(n_taxa)
  n_trees <- length(trees)
  list.encode <- list()

  cat("Computing encoding vectors...\n")
  
  for (n in 1:n_trees){
    progress(n, n_trees, progress.bar = TRUE, init = (n==1))
    tree <- trees[[n]] # extract tree
    tree.encode   <- encode_bisse(tree) # encode the tree
    format.encode <- format_encode(tree.encode, max_taxa) # format the encoding
    list.encode[[n]] <- format.encode # save to list 
  }
  
  # Convert the list of vectors to a torch tensor 
  matrix.encode <- as.data.frame(do.call(cbind, list.encode)) %>% 
    as.matrix() 
  
  cat("\nComputing encoding vectors... Done.\n")

  return(matrix.encode)
  
}


