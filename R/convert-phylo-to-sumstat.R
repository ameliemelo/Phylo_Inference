# This file contains the functions to compute the summary statistics 
# of a phylogenetic tree
# See Saulnier Et. Al. 2017 - PLOS Computational Biology
# DOI:10.1371/journal.pcbi.1005416


#### Nodes data.frame ####


#' Initialize a data.frame for nodes 
#'
#' Create a data.frame filled with NA values, the column of the data.frame
#' are properties of the nodes of the phylogenetic tree. These columns are later
#' filled by the functions fill_xxx_nodes() where 'xxx' is the name of the 
#' column we want to fill.
#'
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
create_nodes_data_frame <- function(tree){
  n <- 2*length(tree$tip.label) - 1 
  n_tips <- length(tree$tip.label)
  df.nodes <- data.frame(index=rep(NA,n),   # index of the node
                         is.tip=rep(NA,n),  # is the node a tip?
                         dist=rep(NA,n),    # distance to the root
                         part=rep(NA,n),    # (3*dist) %/% h + 1 in {1;2;3}
                         depth=rep(NA,n),   # distance the root (in edges)
                         colless=rep(NA,n), # colless score
                         stair=rep(NA,n))   # staircaseness 
  return(df.nodes)
}


#' Fill the "index" column of df.nodes
#'
#' Fill the "index" column of the nodes data.frame for nodes with their indexes. 
#' These indexes correspond to the rows.
#'
#' @param df.nodes data.frame created by create_nodes_data_frame()
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_index <- function(df.nodes){
  n <- length(df.nodes[,1])
  df.nodes["index"]  <- 1:n
  return(df.nodes)
}


#' Fill the "is.tip" column of df.nodes
#'
#' Fill the "is.tip" column of the nodes data.frame with logical. 
#' If the node is a tip, the entry is TRUE
#' If the node is not a tip (i.e. an internal node), the entry is FALSE
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_is.tip <- function(df.nodes, tree){
  n_tips <- length(tree$tip.label)
  df.nodes["is.tip"] <- as.logical(df.nodes["index"] <= n_tips)
  return(df.nodes)
}


#' Fill the "dist" column of df.nodes
#'
#' Fill the "dist" column of the nodes data.frame with the distance to the root 
#' of each nodes. This distance is computed by the function from castor package 
#' get_all_distances_to_root(tree)
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_dist <- function(df.nodes, tree){
  df.nodes["dist"]   <- castor::get_all_distances_to_root(tree)
  return(df.nodes)
}


#' Fill the "part" column of df.nodes
#'
#' Fill the "part" column of the  nodes data.frame with the corresponding part 
#' of the tree. The phylogenetic tree is cut in three equals part in time.
#' If you call D>0 the time length of our tree (maximal distance to the root), 
#' then nodes with distance to the root d: 
#' - d<D/3 belongs to part 1; 
#' - D/3<d<2D/3 belongs to part 2;
#' - d>2D/3 belongs to part 3.
#'
#' @param df.nodes data.frame
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_part <- function(df.nodes){
  dist <- df.nodes$dist 
  height <- get_height(df.nodes)
  greater1 <- as.numeric(dist > height/3)
  greater2 <- as.numeric(dist > as.numeric(2*height/3))
  df.nodes["part"] <- 1 + greater1 + greater2 
  return(df.nodes)
}


#' Fill the "colless" column of df.nodes
#'
#' Fill the "colless" column of the nodes data.frame with the corresponding 
#' colless score. 
#' The colless score is the absolute difference between the number of leaves on
#' the left side and the number of leaves on the right side.
#' This score can be computed only for internal nodes. Thus, rows 
#' corresponding to tips are left with NA.
#' See Saulnier Et. Al. 2017 - DOI:10.1371/journal.pcbi.1005416
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_colless <- function(df.nodes, tree){
  n <- length(df.nodes[,1])
  for (i in 1:n){
    if (df.nodes[i, "is.tip"] == FALSE){ # compute if node is internal
      df.nodes[i, "colless"] <- get_nodes_colless(tree, i) # colless of node i
    }
  }
  return(df.nodes)
}

#' Fill the "stair" column of df.nodes
#'
#' Fill the "stair" column of the nodes data.frame with the corresponding 
#' stairecaseness. 
#' The stairecaseness is the ratio of the minimal number of leaves on a side 
#' over the maximal number of leaves on a side.
#' the left side and the number of leaves on the right side.
#' This score can be computed only for internal nodes. Thus, rows 
#' corresponding to tips are left with NA.
#' Saulnier Et. Al. 2017 - DOI:10.1371/journal.pcbi.1005416
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_stair <- function(df.nodes, tree){
  n <- length(df.nodes[,1])
  for (i in 1:n){
    if (df.nodes[i, "is.tip"] == FALSE){
      df.nodes[i, "stair"] <- get_nodes_staircaseness2(tree, i)
    }
  }
  return(df.nodes)
}


#' Fill the "depth" column of df.nodes
#'
#' Fill the "depth" column of the nodes data.frame with the corresponding depth. 
#' The depth is the minimum number of edges separating the node to the root. 
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_depth <- function(df.nodes, tree){
  dist_edge <- castor::get_all_distances_to_root(tree, as_edge_count=TRUE)
  df.nodes["depth"] <- dist_edge
  return(df.nodes)
}


#' Fill all columns of df.nodes
#'
#' Fill all the columns the nodes data.frame. 
#' This function calls successively the fill_xxx_nodes() functions to fill each 
#' column one by one.
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_nodes_all <- function(df.nodes, tree){
  # Fill all the columns of the nodes data.frame 
  df.nodes <- fill_nodes_index(df.nodes)
  df.nodes <- fill_nodes_is.tip(df.nodes, tree)
  df.nodes <- fill_nodes_dist(df.nodes, tree)
  df.nodes <- fill_nodes_part(df.nodes)
  df.nodes <- fill_nodes_depth(df.nodes, tree)
  df.nodes <- fill_nodes_colless(df.nodes, tree)
  df.nodes <- fill_nodes_stair(df.nodes, tree)
  return(df.nodes)
}


#' Compute the colless score of a node
#'
#' Compute the absolute difference of the number of right and left leaves of an
#' internal node.
#' See Saulnier Et. Al. 2017 - DOI:10.1371/journal.pcbi.1005416
#'
#' @param tree phylo tree
#' @param node integer  
#'
#' @return data.frame
#' @export
#' @examples
get_nodes_colless <- function(tree, node){
  n_leaf_list <- list(0, 0) # to store the number of left & right leaves
  children <- phangorn::Children(tree, node) # both children of the node
  for (i in 1:2){
    leaf   <- phangorn::Descendants(tree, children[i])[[1]]
    n_leaf <- length(leaf) # number of leaves on one side (L or R)
    n_leaf_list[[i]] <- n_leaf  # save the value 
  }
  # Compute the absolute difference
  colless <- abs(n_leaf_list[[1]] - n_leaf_list[[2]])
  return(colless)
}


#' Compute the stairecaseness of a node
#'
#' Compute the ratio between the minumum and maximum number of leaves on a side
#' for an internal node.
#' See Saulnier Et. Al. 2017 - DOI:10.1371/journal.pcbi.1005416
#'
#' @param tree phylo tree
#' @param node integer  
#'
#' @return data.frame
#' @export
#' @examples
get_nodes_staircaseness2 <- function(tree, node){
  n_leaf_list <- c(0, 0) # store the number of left & right leaves
  children <- phangorn::Children(tree, node) # children of the node 
  for (i in 1:2){
    leaf <- phangorn::Descendants(tree, children[i])[[1]]
    n_leaf <- length(leaf) # number of children on one side (L or R)
    n_leaf_list[[i]] <- n_leaf  # save the value 
  }
  # stair = (min(left or right children)/max(left or right children))
  stair <- min(n_leaf_list)/max(n_leaf_list)
  return(stair)
}


#### end ####


#### Edges data.frame ####


#' Initialize a data.frame for edges 
#'
#' Create a data.frame filled with NA values, the column of the data.frame
#' are properties of the edges of the phylogenetic tree. These columns are later
#' filled by the functions fill_xxx_edges() where 'xxx' is the name of the 
#' column we want to fill.
#'
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
create_edges_data_frame <- function(tree){
  n <- length(tree$edge.length)
  df.edges <- data.frame(index=rep(NA,n),   # index of the edge  
                         is.ext=rep(NA,n),  # is the branch external? 
                         length=rep(NA,n),  # length of the branch
                         part=rep(NA,n),    # part of the tree (n/3)
                         node1=rep(NA,n),   # first node (older)
                         node2=rep(NA,n))   # second node (younger)
  return(df.edges)
}


#' Fill the "index" column of df.edges
#'
#' Fill the "index" column of the edges data.frame for nodes with their indexes. 
#' These indexes correspond to the rows.
#'
#' @param df.edges data.frame created by create_nodes_data_frame()
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_index <- function(df.edges){
  n <- length(df.edges[,1])
  df.edges["index"]  <- 1:n
  return(df.edges)
}


#' Fill the "is.ext" column of df.edges
#'
#' Fill the "is.ext" column of the edges data.frame with logical. 
#' If the edge is external (i.e. linked to a tip), the entry is TRUE
#' If the edge is not external, the entry is FALSE
#'
#' @param df.edges data.frame
#' @param df.nodes data.frame
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_is.ext <- function(df.edges, df.nodes){
  df.edges["is.ext"] <- df.nodes[df.edges$node2, "is.tip"]
  return(df.edges)
}

#' Fill the "length" column of df.edges
#'
#' Fill the "length" column of the edges data.frame with their length (time). 
#'
#' @param df.edges data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_length <- function(df.edges, tree){
  df.edges["length"] <- tree$edge.length
  return(df.edges)
}


#' Fill the "part" column of df.edges
#'
#' Fill the "part" column of the edges data.frame with the corresponding part 
#' of the branch. As for the nodes, the tree is sliced into three equal parts. 
#' The part of the branch is determined by the part of younger node.
#' e.g. if the branch goes from a node belonging the part 3 and a node belonging 
#' to part 2 of the tree, then the branch belongs to part 2.  
#'
#' @param df.edges data.frame
#' @param df.nodes data.frame
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_part <- function(df.edges, df.nodes){
  df.edges["part"] <- df.nodes[df.edges$node2, "part"]
  return(df.edges)
}


#' Fill the "node1" & "node2" columns of df.edges
#'
#' Fill the "node1" & "node2" columns of the edges data.frame. with the relevant
#' indexes.
#' "node1" is the older node of the branch 
#' "node2" is the younger node of the branch
#' As time goes, the branch goes from node1 to node2
#'
#' @param df.edges data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_node12 <- function(df.edges, tree){
  df.edges[c("node1","node2")] <- tree$edge
  return(df.edges)
}


#' Fill all columns of df.edges
#'
#' Fill all the columns the edges data.frame. 
#' This function calls successively the fill_xxx_edges() functions to fill each 
#' column one by one.
#'
#' @param df.edges data.frame
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return data.frame
#' @export
#' @examples
fill_edges_all <- function(df.edges, df.nodes, tree){
  # Fill all the columns of the edges data.frame
  df.edges <- fill_edges_index(df.edges)
  df.edges <- fill_edges_length(df.edges, tree)
  df.edges <- fill_edges_node12(df.edges, tree)
  df.edges <- fill_edges_part(df.edges, df.nodes)
  df.edges <- fill_edges_is.ext(df.edges, df.nodes)
  return(df.edges)
}


#### end ####


#### Summary Statistics  ####


#' Compute the branch lengths statistics
#'
#' Given a list of branch lengths, returns the mean, median and variance.
#'
#' @param bl list
#'
#' @return stats (list) w/ stats$mean, stats$med, stats$var 
#' @export
#' @examples
get_bl_stats <- function(bl){
  stats <- list("mean"=-999, "med"=-999, "var"=-999)
  stats$mean <- mean(bl)
  stats$med  <- median(bl)
  stats$var  <- var(bl)
  return(stats)
}


#' Compute the branch lengths summary statistics 
#'
#' Compute the 24 summary statistics related to branch lengths 
#' - a.stat means that all branches are taken into account to compute stat
#' - e.stat means that only external branches are taken into account 
#' - i.x.stat means that only the internal branches belonging to the part x of 
#'   the tree are taken into account 
#' - ie.x.stat = i.x.stat / e.stat
#' 
#'
#' @param df.edges data.frame
#'
#' @return list
#' @export
#' @examples
get_bl_ss <- function(df.edges){

  ss.bl   <-   list("a.mean"   =-999, "a.med"   =-999, "a.var"   =-999,
                    "e.mean"   =-999, "e.med"   =-999, "e.var"   =-999,
                    "i.1.mean" =-999, "i.1.med" =-999, "i.1.var" =-999,
                    "i.2.mean" =-999, "i.2.med" =-999, "i.2.var" =-999,
                    "i.3.mean" =-999, "i.3.med" =-999, "i.3.var" =-999,
                    "ie.1.mean"=-999, "ie.1.med"=-999, "ie.1.var"=-999,
                    "ie.2.mean"=-999, "ie.2.med"=-999, "ie.2.var"=-999,
                    "ie.3.mean"=-999, "ie.3.med"=-999, "ie.3.var"=-999)
  
  df.edges.ext <- df.edges[df.edges["is.ext"]==TRUE,]
  df.edges.int <- df.edges[df.edges["is.ext"]==FALSE,]
  
  bl.a <- df.edges$length
  bl.e <- df.edges.ext$length
  bl.i.1 <- df.edges.int[df.edges.int["part"]==1,]$length
  bl.i.2 <- df.edges.int[df.edges.int["part"]==2,]$length
  bl.i.3 <- df.edges.int[df.edges.int["part"]==3,]$length
  
  bl.list <- list(bl.a, bl.e, bl.i.1, bl.i.2, bl.i.3)
  n  <- length(bl.list)
  
  for (i in 1:n){
    bl <- bl.list[[i]]
    stats <- get_bl_stats(bl)
    ss.bl[[3*(i-1)+1]] <- stats$mean
    ss.bl[[3*(i-1)+2]] <- stats$med
    ss.bl[[3*(i-1)+3]] <- stats$var
  }
  
  for (i in 1:3){
    for (stat in c("mean", "med", "var")){
      name.i  <- paste("i" , i, stat, sep=".")
      name.ie <- paste("ie", i, stat, sep=".")
      name.e <- paste("e", stat, sep=".")
      stat.e <- ss.bl[[name.e]]
      stat.i <- ss.bl[[name.i]]
      ratio <- stat.i / stat.e
      ss.bl[[name.ie]] <- ratio
    }
  }
  return(ss.bl)
}


#' Check that is the node is in a ladder 
#'
#' A node is a considered to be in a ladder if:
#' one of its children is a tip & the other one is an internal node.
#'
#' @param node int 
#' @param tree phylo tree
#'
#' @return logical
#' @export
#' @examples
is_in_ladder <- function(node, tree){
  sum <- 0
  if (!is_tip(node, tree)){
    children <- phangorn::Children(tree, node)
    sum <- as.numeric(is_tip(children[1], tree) + is_tip(children[2], tree))  
  }
  return(sum==1)
}


#' Traverse the tree and write ladders
#'
#' Traverse the tree in preoder and write the nodes indexes belonging to ladders 
#' found through the traverse. Nodes that don't belong to ladder are marked
#' as -1.
#'
#' @param tree phylo tree
#' @param node integer 
#' @param preorder vector 
#'
#' @return vector containing ladders (separated by -1)
#' @export
#' @examples
traverse_ladder <- function(tree, node, preorder){
  n_tips <- length(tree$tip.label)
  root <- n_tips + 1 
  if (is_in_ladder(node, tree)){
    preorder <- c(preorder, node)
  }
  else{
    preorder <- c(preorder, -1)
  }
  children <- get_child(node, tree)
  child.left <- children$right
  child.right <- children$left
  if (!is.na(child.left)){
    preorder <- traverse_ladder(tree, child.left, preorder)
  }
  if (!is.na(child.right)){
    preorder <- traverse_ladder(tree, child.right, preorder)
  }
  return(preorder)
}


#' Extract ladders from vector 
#'
#' Take the vector generated by traverse_ladder() and extract the ladders 
#' from it. To do so, we append each ladder (as a sequence of nodes index) to 
#' a list. Ladder of size one are deleted. 
#'
#' @param traverse_ladder vector generate by traverse_ladder()
#'
#' @return list of ladders 
#' @export
#' @examples
extract_ladders <- function(traverse_ladder){
  n <- length(traverse_ladder)
  ladder.list <- list()
  ladder  <- c()
  for (i in 1:n){
    x <- traverse_ladder[i]
    if (x != -1){ladder <- c(ladder, x)}
    else if (x==-1 & length(ladder)==1){ladder <- c()}
    else if (x==-1 & length(ladder)>=2){
      ladder.list <- append(ladder.list, list(ladder))
      ladder <- c()
    }
  }
  return(ladder.list)
}


#' Sum up the process to find ladders in a tree
#'
#' @param tree phylo tree
#'
#' @return list of ladders 
#' @export
#' @examples
get_ladders <- function(tree){
  root <- length(tree$tip.label) + 1 
  traverse_ladder <- traverse_ladder(tree, root, c())
  ladders <- extract_ladders(traverse_ladder)
  return(ladders)
}


#' Get the maximum ladder size of the tree (divided by n_tips)
#'
#' Compute the maximum ladder size (in number of nodes) and divide by the number
#' of tips in the tree
#'
#' @param tree phylo tree
#'
#' @return numeric
#' @export
#' @examples
get_max_ladder <- function(tree){
  max_l <- 0
  n_tips <- length(tree$tip.label)
  ladders <- get_ladders(tree)
  for (ladder in ladders){
    l <- length(ladder)
    if (l > max_l){max_l <- l}
  }
  return(max_l / n_tips)
}

#' Get the the proportion of internal nodes in ladder
#'
#' Compute the proportion of internal nodes that belongs to a ladder. Ladder of 
#' size one (composed of a single node) are counted as ladder. 
#'
#' @param tree phylo tree
#'
#' @return numeric
#' @export
#' @examples
get_il_nodes <- function(tree){
  n_tips <- length(tree$tip.label)
  n_nodes <- n_tips - 1 
  ladders <- get_ladders(tree)
  il_nodes <- 0
  for (ladder in ladders){
    l <- length(ladder)
    il_nodes <- il_nodes + l 
  }
  return(il_nodes / n_nodes)
}


#' Get the proportion of imbalanced nodes 
#'
#' An imbalanced nodes is an internal node that has a different number of leaves 
#' on the right size and on the left size 
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return numeric
#' @export
#' @examples
get_staircaseness1 <- function(df.nodes, tree){
  n <- 2*length(tree$tip.label) - 1 
  n_tips <- length(tree$tip.label)
  n_nodes <- n - n_tips
  stair1 <- table(df.nodes[df.nodes["stair"]==1,]$stair)[[1]] / n_nodes 
  return(1 - stair1)
}


#' Get all the topological summary statistics 
#'
#' Summarize all the calls needed to have the summary statistics related to the 
#' tree topology 
#'
#' @param df.nodes data.frame
#' @param tree phylo tree
#'
#' @return list (of topological summary statistics)
#' 
#' @export
#' @examples
get_topo_ss <- function(df.nodes, tree){

  ss.topo <- list("height"  = -999, "colless" = -999, "sackin"     = -999, 
                  "wdratio" = -999, "deltaw"  = -999, "max_ladder" = -999,
                  "il_nodes"= -999, 
                  "stair1"  = -999, "stair2"     = -999)
  
  ss.topo$height     <- get_height(df.nodes)
  ss.topo$colless    <- sum(df.nodes$colless, na.rm = TRUE)
  ss.topo$sackin     <- sum(df.nodes[df.nodes["is.tip"]==TRUE,]$depth)
  ss.topo$wdratio    <- get_wdratio(df.nodes) 
  ss.topo$deltaw     <- get_deltaw(df.nodes)
  ss.topo$max_ladder <- get_max_ladder(tree)
  ss.topo$il_nodes   <- get_il_nodes(tree)
  ss.topo$stair1     <- get_staircaseness1(df.nodes, tree) 
  ss.topo$stair2     <- sum(df.nodes$stair, na.rm = TRUE)
  return(ss.topo)
}


#' Get the tree height 
#'
#' The height of the tree is also the distance to the root of each tip
#'
#' @param df.nodes data.frame
#'
#' @return numeric 
#' @export
#' @examples
get_height <- function(df.nodes){
  height <- max(df.nodes$dist)
  return(height)
}


#' Get the wdratio of the tree
#'
#' wdratio is defined as the maximal width over the maximal depth of a tree
#' the width is the number of nodes at a given depth
#'
#' @param df.nodes data.frame
#'
#' @return numeric 
#' @export
#' @examples
get_wdratio <- function(df.nodes){
  width <- max(table(df.nodes$depth))
  depth <- max(df.nodes$depth)
  ratio <- width/depth
  return(ratio)
}


#' Get the deltaw of the tree
#'
#' deltaw is defined as the maximal difference in width between two consecutive 
#' depths 
#'
#' @param df.nodes data.frame
#'
#' @return numeric 
#' @export
#' @examples
get_deltaw <- function(df.nodes){
  table.width <- table(df.nodes$depth)
  n <- length(table.width)-1
  diff <- 0
  for (i in 1:n){
    diff <- max(diff, abs(table.width[[i]] - table.width[[i+1]]))
  }
  return(diff)
}


#' Get sampled coordinates of the LTT  
#'
#' We take 20 points uniformly distributed in N (number of lineages). 
#' We use these 20 points to sample the LTT
#' Not that for this statistics to be relevant the tree needs to have at least 
#' 20 lineages.
#'
#' @param tree phylo tree
#'
#' @return matrix 
#' @export
#' @examples
get_ltt_coords <- function(tree){
  ltt.coord <- ape::ltt.plot.coords(tree) 
  n_events <- length(ltt.coord[,1])
  step <- n_events %/% 20
  bins <- c()
  for (i in 0:19){
    bins <- c(bins, 1 + i*step)
  }
  out.ltt.coord <- ltt.coord[bins,]
  coord <- convert_coord_to_list(out.ltt.coord)
  return(coord)
}


#' Get slopes of the LTT
#'
#' Divide the tree in 3 equals part (trough time) and compute the slope for each
#' part (named slope$p.i). Then do the ratio of:
#' - slope of the 2nd over the slope of the 1st part 
#' - slope of the 3rd over the slope of the 2nd part 
#'
#' @param tree phylo tree 
#' @param n integer saying in how many slices we want to cut our time interval
#'
#' @return list (of slopes and ratio)
#' @export
#' @examples
get_ltt_slopes <- function(tree, n=10){
  
  slopes <- list()
  
  ltt.coord <- ape::ltt.plot.coords(tree) 
  ltt.df <- as.data.frame(ltt.coord)
  ltt.df["N"] <- log(ltt.df["N"])
  ltt.df.list <- divide_ltt.df_points(ltt.df, n)
  if (!has_an_empty_df(ltt.df.list)){
    for (i in 1:n){
      ltt.df.i <- ltt.df.list[[paste("part", i, sep=".")]]
      slopes[[paste("slope", i, sep=".")]] <- lm(N ~ time, ltt.df.i)$coef[[2]]
    }
  }  
  else{for (i in 1:n){slopes[[paste("slope", i, sep=".")]] <- NA}}
  return(slopes)
}


#' Check if the list has an empty data.frame
#'
#' 
#' @param l list containing data.frames 
#'
#' @return logical: if l has an empty df, returns TRUE; else returns FALSE 
#' @export
#' @examples
has_an_empty_df <- function(l){
  bool <- FALSE
  for (df in l){
    if (nrow(df) == 0){bool <- TRUE}
  }
  return(bool)
}


#' Divide ltt.df into n sub-data.frame of equal time interval
#'
#' Let's consider a tree of root time T<0 and t=0 is the present. 
#' divide_ltt.df returns a list of n data.frame 
#' the i data.frame of the list containes ltt coordinates such that the time (t)
#' of these coordinates verifies: 
#' (n+1-i)/n <= t <= (n-i)/n
#'
#' @param ltt.df data.frame containing the ltt coordinates 
#' @param n integer saying in how many slices we want to split ltt.df
#'
#' @return ltt.df.list $i contains the i data.frame 
#' @export
#' @examples
divide_ltt.df_time <- function(ltt.df, n){
  ltt.df.list <- list() # list storing the parts of the data.frame
  t <- ltt.df$time[1] # root time
  for (i in 1:n){
    inf <- (n + 1 - i)*t/n  # lower time bound (>= inf)
    sup <- (n - i)*t/n      # upper time bound (<= sup)
    ltt.df.i <- ltt.df[ltt.df["time"] >= inf & ltt.df["time"] <= sup,] 
    ltt.df.list[[paste("part", i, sep=".")]] <- ltt.df.i
  }
  return(ltt.df.list)
}


divide_ltt.df_points <- function(ltt.df, n){
  ltt.df.list <- list()
  len <- nrow(ltt.df)
  bounds <- as.integer(seq(1, len, length.out=n+1))
  bounds[n+1] <- bounds[n+1] 
  for (i in 1:n){
    inf <- bounds[i] + (i!=1)
    sup <- bounds[i+1]
    ltt.df.i <- ltt.df[inf:sup,]
    ltt.df.list[[paste("part", i, sep=".")]] <- ltt.df.i
  }
  return(ltt.df.list)
}


#' Compute all the summary statistics 
#'
#'
#' @param df.nodes data.frame
#' @param df.edges data.frame
#' @param tree phylo tree 
#'
#' @return list (of topological summary statistics)
#' @export
#' @examples
get_all_ss <- function(df.nodes, df.edges, tree){
  ss.bl   <- get_bl_ss(df.edges)                 # branch length ss 
  ss.topo <- get_topo_ss(df.nodes, tree)         # topological ss
  ss.ltt.coord <- get_ltt_coords(tree)
  ss.ltt.slopes <- get_ltt_slopes(tree)
  
  ss.all <- c(ss.bl, list("n_taxa" = ceiling(nrow(df.nodes)/2)),
              ss.topo, ss.ltt.slopes, ss.ltt.coord)    # merge all ss in a single list 
  return(ss.all)
}


#' Convert the matrix of LTT coordinates into a list
#'
#'
#' @param ltt.coord matrix containing LTT coordinates 
#'
#' @return list (of LTT coordinates)
#' @export
#' @examples
convert_coord_to_list <- function(ltt.coord){
  l <- list() # initialize the list 
  ltt.names <- create_ltt.names.coord() # get the names of the LTT coord.
  for (i in 1:40){ # 40 coordinates (20 for x coord. + 20 for y coord)
    row <- 1 + (i-1)%%20
    col <- 1 + (i-1)%/%20
    l[ltt.names[i]] <- ltt.coord[row, col]
  }
  return(l)
}


#' Sum up the process to compute all the summary statistics in one function 
#'
#'
#' @param tree phylo tree 
#'
#' @return list (of topological summary statistics)
#' @export
#' @examples
get_ss <- function(tree){
  
  # Create nodes data.frame
  df.nodes <- create_nodes_data_frame(tree)
  df.nodes <- fill_nodes_all(df.nodes, tree)
  
  # Create edges data.frame
  df.edges <- create_edges_data_frame(tree)
  df.edges <- fill_edges_all(df.edges, df.nodes, tree)
  
  # Compute all summary statistics 
  ss <- get_all_ss(df.nodes, df.edges, tree)
  return(ss)
}


#### end ####


#### Format Summary Statistics #####


#' Create the vector containing the ltt statistic names
#'
#'
#' @param void 
#'
#' @return vector
#' @export
#' @examples
create_ltt.names.coord <- function(){
  ltt.names.coord <- c() # initialize vector 
  for (coord in c("t", "N")){ # x & y coordinates 
    for (i in 1:20){ # 20 points for each
      new_name <- paste("ltt", coord, i, sep="_")
      ltt.names.coord <- c(ltt.names.coord, new_name)
    }
  }
  return(ltt.names.coord)
}


#' Create the vector containing the ltt statistic names
#'
#'
#' @param void 
#'
#' @return vector
#' @export
#' @examples
create_ltt.names.slope <- function(n=10){
  ltt.names.slope <- c() # initialize vector 
  for (i in 1:n){ # 20 points for each
      new_name <- paste("ltt", "slope", i, sep="_")
      ltt.names.slope <- c(ltt.names.slope, new_name)
  }
  return(ltt.names.slope)
}


create_ltt.names.topo <- function(){
  
  topo.names <- c("height", "colless", "sackin",
                  "wdratio", "deltaw", "max_ladder", 
                  "il_nodes", "stair1", "stair2")
  
  return(topo.names)
}

#' Create the vector containing the summary statistic names
#'
#'
#' @param void 
#'
#' @return vector
#' @export
#' @examples
create_ss.names <- function(){
  ss.names <- c("a.mean", "a.med", "a.var", 
                "e.mean", "e.med", "e.var", 
                "i.1.mean", "i.1.med", "i.1.var",
                "i.2.mean", "i.2.med", "i.2.var",
                "i.3.mean", "i.3.med", "i.3.var",
                "ie.1.mean", "ie.1.med", "ie.1.var",
                "ie.2.mean", "ie.2.med", "ie.2.var",
                "ie.3.mean", "ie.3.med", "ie.3.var", "n_taxa")
  
  ltt.names.topo  <- create_ltt.names.topo()
  ltt.names.slope <- create_ltt.names.slope()
  ltt.names.coord <- create_ltt.names.coord()
  ss.names <- c(ss.names, ltt.names.topo, ltt.names.slope,
                ltt.names.coord)
  
  return(ss.names)
}


#' Create an empty data.frame to store the summary statistics
#'
#'
#' @param ss.names vector of summary statistic names 
#'
#' @return data.frame (nrow=0, ncol=length(ss.names))
#' @export
#' @examples
create_ss_dataframe <- function(ss.names){
  df <- data.frame(matrix(nrow = 0, ncol = length(ss.names)))
  colnames(df) <- ss.names
  return(df)
}


#' Add a row to the summary statistics data.frame
#'
#' Each row on the data.frame corresponds to the summary of a tree.
#' In short, one row = one tree, one column = one summary statistics
#'
#' @param ss.names vector of summary statistic names 
#'
#' @return data.frame (nrow=0, ncol=length(ss.names))
#' @export
#' @examples
add_row <- function(df, ss){
  ss.val <- unlist(ss, use.names = FALSE)
  df[nrow(df) + 1, ] <-  ss.val
  return(df)
}


#' Generate a summary statistics data.frame
#'
#' Create and fill a data.frame storing summary statistics of several trees.
#'
#' @param n_trees number of trees we want to generate 
#' @param n_taxa number of taxa of each tree
#' @param lambda_min minimum speciation rate 
#' @param lambda_max maximum speciation rate
#' @param mu_min minimum extinction rate 
#' @param mu_max maximum extinction rate
#'
#' @return data.frame (nrow=n_trees, ncol=length(ss.names))
#' @export
#' @examples
generate_ss_dataframe <- function(n_trees, n_taxa, 
                                  lambda_min, lambda_max, 
                                  mu_min, mu_max){
  ss.names <- create_ss.names()
  df <- create_ss_dataframe(ss.names) # initialize the data.frame (empty)
  lambda_vec <- c()
  mu_vec <- c()
  r_vec <- c()
  epsilon_vec <- c()
  while (nrow(df) < n_trees){
    lambda <- runif(1, lambda_min, lambda_max) # generate a random spec. rate
    mu <- runif(1, mu_min, mu_max) # same w/ ext. rate 
    tree <- trees(c(lambda, mu), "bd", max.taxa=n_taxa)[[1]] # create tree (BD)
    ss <- get_ss(tree) # compute summary statistics 
    if (!any(is.na(ss))){
      df <- add_row(df, ss)
      lambda_vec <- c(lambda_vec, lambda)
      mu_vec <- c(mu_vec, mu)
      r_vec <- c(r_vec, lambda - mu)
      epsilon_vec <- c(epsilon_vec, mu/lambda)
      progress(100*nrow(df)/n_trees)
      }
  }
  df$lambda <- lambda_vec
  df$mu <- mu_vec
  df$r <- r_vec
  df$epsilon <- epsilon_vec
  return(df)
}


generateSumStatFromPhylo <- function(trees, true.param){
  
  ss.names <- create_ss.names() # summary statistic names 
  df <- create_ss_dataframe(ss.names) # initialize the data.frame (empty)
  
  n_trees <- length(trees)
  
  cat("Computing tree summary statistics...\n")

  for (i in 1:n_trees){
    tree <- trees[[i]] # get the tree
    ss   <- get_ss(tree) # compute summary statistics 
    df <- add_row(df, ss)
    progress(i, n_trees, progress.bar = TRUE, init = (i==1)) # print progress 
  }
  
  cat("\nComputing tree summary statistics... Done.")
  
  names.param <- names(true.param) # names of the parameters (e.g. "lambda" "mu")
  for (name in names.param){
    param.vec  <- true.param[[name]]
    df[name] <- param.vec # store parameter values in data.frame
  }
  
  return(df)
}



scale_summary_statistics <- function(df, n_taxa, name.param){
  col.ltt.t <- colnames(df)[substring(colnames(df),1,5) == "ltt_t"]
  col.ltt.n <- colnames(df)[substring(colnames(df),1,5) == "ltt_N"]
  col.ss    <- !(colnames(df) %in% c(col.ltt.t, col.ltt.n, name.param))
  
  df[col.ltt.t] <- df[col.ltt.t] / abs(min(df[col.ltt.t]))
  df[col.ltt.n] <- df[col.ltt.n] / max(n_taxa)
  df[col.ss]    <- scale(df[col.ss], center = FALSE, scale = TRUE)
  
  #col.to.scale <- c("sackin", "colless", "stair2")
  #df[col.to.scale]    <- scale(df[col.to.scale],
  #                             center = FALSE, scale = TRUE)

  
  return(df)
}


#' Convert a data.frame to a torch::dataset 
#'
#'
#' @param df data.frame to convert 
#' @param direct_target logical (default=FALSE)
#' If TRUE, targets are (lambda, mu)
#' If FALSE, targers are (r=lambda-mu, epsilon=mu/lambda)
#'
#' @return torch::dataset()
#' @export
#' @examples
convert_ss_dataframe_to_dataset <- function(df){
  
  ss_dataset <- torch::dataset(
    
    name <- "summary_statistics_dataset", 
    
    initialize = function(df, name.param, exclude.param = ""){

      dataframe <- na.omit(df) # delete NA
      
      # input data
      topo.names <- c("height", "colless", "sackin", "wdratio", "deltaw", "max_ladder",
                      "il_nodes", "stair1", "stair2")
      
      #for (name in c("sackin", "colless")){df[name] <- 0}
      
      x = df[!(colnames(df) %in% c(name.param, exclude.param))] %>% 
        as.matrix()
      self$x <- torch_tensor(x)
      
      # target data 
      y = df[name.param] %>% 
        as.matrix()
      self$y <- torch_tensor(y)
  
    }, 
    
    .getitem = function(i) {
      list(x = self$x[i, ], y = self$y[i, ])
    }, 
    
    .length = function() {
      self$y$size()[[1]]
    }
    
  )
  return(ss_dataset)
}


#### end ####


if (FALSE){ # don't execute the following when imported with 'source'
  
  
  tree <- trees(c(.1, 0.05), "bd", max.taxa=500)[[1]]
  plot(tree)
  nodelabels()
  tiplabels()
  
  tree$orig
  
  lambda_vec <- c()
  height_vec <- c()
  
  n <- 10
  tree <- trees(c(.1, 0.05), "bd", max.taxa=1000)[[1]]
  slopes <- get_ltt_slopes(tree, n)
  plot(1:n, slopes, ylim = c(.04,.11))
  abline(.1,0)
  abline(.05,0)
  
  for (i in 1:300){
    lambda <- runif(1, 0.0, 0.5)
    lambda_vec <- c(lambda_vec, lambda)
    tree <- trees(c(lambda, 0), "bd", max.taxa=50)[[1]]
    df.nodes <- create_nodes_data_frame(tree)
    df.nodes <- fill_nodes_all(df.nodes, tree)
    height <- get_height(df.nodes)
    #height <- max(castor::get_all_distances_to_root(tree))
    height_vec <- c(height_vec, height)
  }
  
  plot
  
  root <- length(tree$tip.label) + 1 
  traverse_ladder <- traverse_ladder(tree, root, c())
  ladders <- get_ladders(traverse_ladder)
  for (l in ladders){
    print(length(l))
  }
  
  
  df.nodes
  
  ss <- get_ss(tree)
  ss$coord 
  data.frame("coord" = ss$coord, "x" = 0)
  
  
  df.nodes <- create_nodes_data_frame(tree)
  df.nodes <- fill_nodes_all(df.nodes, tree)
  df.edges <- create_edges_data_frame(tree)
  df.edges <- fill_edges_all(df.edges, df.nodes, tree)
  
}


