# This file contains the general functions needed during the interference 
# processes 

#### Operations on Rates ####

#' Compute r and epsilon
#'
#' Compute diversification rate (r) and turnover rate (epsilon) given the 
#' speciation rate (lambda) and extinction rate (lambda). 
#' Formally : 
#' r = lambda - mu
#' epsilon = mu / lambda
#'
#' @param vec.lambda_mu vector of length 2 containing lambda and mu
#'
#' @return vec.r_epsilon vector of length 2 containing r and epsilon 
#' @export
#' @examples
get_r_epsilon_single <- function(vec.lambda_mu){
  lambda  <- vec.lambda_mu[1]
  mu      <- vec.lambda_mu[2]
  r       <- lambda - mu
  epsilon <- mu / lambda
  vec.r_epsilon <- c(r, epsilon)
  return(vec.r_epsilon)
}


#' Compute lambda and mu 
#'
#' Compute speciation rate (lambda) and extinction rate (mu) given the 
#' diversification rate (r) and turnover rate (epsilon). 
#' Formally : 
#' lambda  =            r / (1 - epsilon)
#' mu      = epsilon *  r / (1 - epsilon)
#'
#' @param vec.r_epsilon vector of length 2 containing r and epsilon
#'
#' @return vec.lambda_mu vector of length 2 containing lambda and mu 
#' @export
#' @examples
get_lambda_mu_single <- function(vec.r_epsilon){
  r       <- vec.r_epsilon[1]
  epsilon <- vec.r_epsilon[2]
  lambda  <- r / (1 - epsilon)
  mu      <- epsilon * lambda 
  vec.lambda_mu <- c(lambda, mu)
  return(vec.lambda_mu)
}


#' Extend get_r_epsilon_single to vectors  
#'
#' Given a vector of speciation rates and a vector of extinction rates
#' compute the corresponding diversification rates and turnover rates
#'
#' @param list.lambda_mu $lambda vector of speciation rates 
#'                       $mu     vector of extinction rates
#'
#' @return list.r_epsilon $r        vector of diversification rates
#'                        $epsilon  vector of turnover rates 
#' @export
#' @examples
get_r_epsilon_extended <- function(list.lambda_mu){
  vec.lambda  <- list.lambda_mu$lambda
  vec.mu      <- list.lambda_mu$mu
  vec.r       <- c()
  vec.epsilon <- c()
  n           <- length(vec.mu)
  
  for (i in 1:n){
    vec.lambda_mu <- c(vec.lambda[i], vec.mu[i])
    vec.r_epsilon <- get_r_epsilon_single(vec.lambda_mu)
    r <- vec.r_epsilon[1]
    epsilon <- vec.r_epsilon[2]
    vec.r <- c(vec.r, r)
    vec.epsilon <- c(vec.epsilon, epsilon)
  }
  
  list.r_epsilon <- list("r" = vec.r, "epsilon" = vec.epsilon)
  
  return(list.r_epsilon)
}


#' Extend get_lambda_mu_single to vectors  
#'
#' Given a vector of diversification rates and a vector of turnover rates
#' compute the corresponding speciation rates and extinction rates
#'
#' @param list.r_epsilon $r        vector of diversification rates
#'                        $epsilon  vector of turnover rates 
#'
#' @return list.lambda_mu $lambda vector of speciation rates 
#'                       $mu     vector of extinction rates
#' @export
#' @examples
get_lambda_mu_list <- function(list.r_epsilon){
  vec.r       <- list.r_epsilon$r
  vec.epsilon <- list.r_epsilon$epsilon
  vec.lambda  <- c()
  vec.mu      <- c()
  n           <- length(vec.r)
  
  for (i in 1:n){
    vec.r_epsilon <- c(vec.r[i], vec.epsilon[i])
    vec.lambda_mu <- get_lambda_mu_single(vec.r_epsilon)
    lambda <- vec.lambda_mu[1]
    mu     <- vec.lambda_mu[2]
    vec.lambda <- c(vec.lambda, lambda)
    vec.mu     <- c(vec.mu, mu)
  }
  
  list.lambda_mu <- list("lambda" = vec.lambda, "mu" = vec.mu)
  
  return(list.lambda_mu)
}


#### end ####


#### Plotting ####

#' Plot predictions vs. truth
#'
#' Draw a scatter plot represented the predicted values by a model (e.g. DNN) 
#' vs. the true values.
#'
#' @param pred.list list of vectors of predicted paremeters 
#'                  $lambda is the vector of predicted values for lambda
#'                  $mu is the vector of predicted values for mu
#' @param true.list list of vectors of corresponding true values 
#'                  $lambda is the vector of true values for lambda
#'                  $mu is the vector of true values for mu
#' @param save logical, should the plot be saved? (default = FALSE)
#' @param fname if save = TRUE, name of the saved file (default = NA)
#' @param r2_score logical, should the r2 score be printed in the plot? 
#'                 (default = TRUE)
#' @param mse_score logical, should the MSE be printed in the plot?
#'                  (default = TRUE)
#' @param lm_fit logical, should the scatter be fitted by a linear regression?
#'               (default = TRUE)
#' @param subtitles subtitle to print below plots (default = "")
#' @param alone logical, is this function called to draw only one plot? 
#'              (default = TRUE)
#'              
#' @return 
#' 
#' @export
#' @examples
plot_pred_vs_true <- function(pred.list, true.list, names, save = FALSE, fname = NA,
                              r2_score = TRUE, mse_score = TRUE, 
                              lm_fit = TRUE, subtitles = "", alone = TRUE, 
                              bar = FALSE){
  
  n <- length(pred.list)
  n_row <- n %/% 2
  n_col <- 2 + as.integer(bar)
  
  if (bar){
    name <- paste(subtitles[[1]], names[[1]], sep = ".")
    df.error.decomp <- data.frame(c(NA, NA))
    colnames(df.error.decomp) <- name
    }
  
  lambda.lim <- c(0.05, 1.05) 
  mu.lim <- c(0.,0.9)
  lim <- list(lambda.lim, mu.lim)
  
  if (save){
    fname <- paste(fname, "pdf", sep=".")
    aspect_ratio <- 1.62 * (n_col / n_row)
    pdf(fname, width = 10, height = 10/aspect_ratio, pointsize = 15/sqrt(aspect_ratio))
    }
  
  if (alone){par(mfrow=c(n_row,n_col))}
  
  for (i in 1:n){
    
    #pred.name <- paste(names[[1 + (i-1)%%2]], method, sep=".") # parameter name
    pred <- pred.list[[i]] # parameter prediction 
    true <- true.list[[1 + (i-1)%%2]] # parameter true 
    
    # Evaluate the R2 score of predictions vs. truth
    if (r2_score){
      r2 <- R2_Score(pred, true) # compute r2
      r2 <- format(round(r2, 3), nsmall = 3) # format r2 
    }
    
    # Evaluate the MSE of predictions vs. truth 
    if (mse_score){
      mse <- MSE(pred, true) # compute mse 
      mse <- format(round(mse, 3), nsmall = 3) # format mse
    }
    
    rate_name <- names[[1 + (i-1)%%2]]
    main <- create_main_plot_pred_vs_true(rate_name, r2_score, mse_score, r2, mse)
    
    # Scatter plot - Predictions vs. truth
    plot(true, pred, main = main,
         sub = subtitles[1 + (i-1)%/%2], 
         xlim = lim[[1 + (i-1)%%2]], ylim = lim[[1 + (i-1)%%2]])

    # Identity line (for the eye)
    abline(0, 1) # x -> y line (for the eye)
    
    # Linear fit of predictions vs. truth (to see significant trends)
    if (lm_fit){
      fit <- lm(pred ~ true)
      fit.scale = lm(pred - true ~ scale(true, scale = FALSE))
      sig = summary(fit)$coefficients[2,4]
      sig.scale = summary(fit.scale)$coefficients[2,4]
      coef <- as.numeric(summary(fit.scale)$coefficients[, 1])
      p_intercept <- as.numeric(summary(fit.scale)$coefficients[1, 4])
      p_slope     <- as.numeric(summary(fit.scale)$coefficients[2, 4])
      sig_intercept <- get_significant_code(p_intercept)
      sig_slope     <- get_significant_code(p_slope)
      intercept <- round(coef[1], 3) 
      slope     <- round(coef[2], 3) 
      text      <- TeX(paste("$\\", slope, "^{", sig_slope, "} \\cdot x + ",
                             intercept, "^{", sig_intercept, sep = ""))
      abline(fit, col="red", lty = ifelse(sig < .05,1,2))
      mtext(text, line = -2, adj = 0)
    }
    
    if (bar){
      a <- fit$coefficients[[1]]
      b <- fit$coefficients[[2]]
      true.lm <- a + b*pred
      var  <- 1 - R2_Score(true.lm, true)
      err  <- 1 - R2_Score(pred, true)
      bias <- err - var
      name <- paste(subtitles[1 + (i-1)%/%2], names[[1 + (i-1)%%2]], sep=".")
      df.error.decomp[name] <- c(var, bias)
    }
  }
  
  if (bar){
    rownames(df.error.decomp) <- c("Variance", "Bias")
    barplot(as.matrix(df.error.decomp), col = c("blue", "purple"), ylab = "Error")
    legend("topright",                                   
           legend = c("Variance", "Bias"),
           fill = c("blue", "purple"))
  }
  
  if (save){dev.off()}
}


plot_pred_vs_true_single <- function(pred, true, name, range.out, range.in,
                                     main = '', lm = TRUE, print_mse = TRUE, 
                                     print_r2 = FALSE){
  
  #print(main)
  range.size <- range.out[2] - range.out[1]
  marge <- 0.2 * range.size
  xy.lim <- c(range.out[1] - marge, range.out[2] + marge)
  filter_param <- filter_preds_boundary(pred, true, range.in[1], range.in[2])
  pred.in  <- filter_param$pred.in
  true.in  <- filter_param$true.in
  pred.out <- filter_param$pred.out
  true.out <- filter_param$true.out
  mse <- ifelse(print_mse, round(RMSE(pred.in, true.in)/mean(true.in),3), NA)
  r2  <- ifelse(print_r2, round(R2_Score(pred.in, true.in),3), NA)
  sub <- create_main_plot_pred_vs_true(name, print_r2, print_mse, r2, mse)
  sub <- ""
  lab.pred <- TeX(paste("$\\", name, "_{,pred}$", sep=""))
  lab.true <- TeX(paste("$\\", name, "_{,true}$", sep=""))
  plot(true.in, pred.in, main = main, sub = sub, xlim = xy.lim, ylim = xy.lim,
       cex = .6, col = alpha("black", .7), xlab = lab.true, ylab = lab.pred)
  points(true.out, pred.out, cex = .6, col = alpha("darkorange4", .7))
  abline(0,1)
  if (lm){
    fit <- lm(pred.in ~ true.in)
    sig = summary(fit)$coefficients[2,4]
    abline(fit, col="red", lty = ifelse(sig < .05,1,2))
  }
}


plot_pred_vs_true_all <- function(pred.param, true.param, name.param, param.range.out,
                                  param.range.in,
                                  save = FALSE, fname = NA, lm = TRUE,
                                  print_mse = TRUE, print_r2 = FALSE){
  
  
  
  name.model.tex <- list("mle" = TeX("MLE"), "dnn_ss" = TeX("$DNN_{SS}$"), 
                         "cnn_ltt" = TeX("$CNN_{LTT}$"),
                         "rnn_ltt" = TeX("$RNN_{LTT}$"),
                         "cnn_cblv" = TeX("$CNN_{CBLV}$"),
                         "gnn_phylo" = TeX("$GNN_{Phylo}$"),
                         "cnn-ltt" = TeX("$CNN_{LTT}$"), "dnn-ss" = TeX("$DNN_{SS}$"),
                         "rnn-ltt" = TeX("$RNN_{LTT}$"),
                         "cnn-cblv" = TeX("$CNN_{CBLV}$"),
                         "cnn_cblv_notips" = TeX("$CNN_{CBLV}^{notips}$"),
                         "cnn_cblv_disotips" = TeX("$CNN_{CBLV}^{disotips}$"),
                         "gnn-phylo" = TeX("$GNN_{Phylo}$"))
  
  n_model <- length(pred.param)
  name.model <- names(pred.param)
  n_param <- length(pred.param[[1]])
  
  par(mfrow = c(n_model, n_param))

  if (save){
    fname <- paste(fname, "pdf", sep=".")
    aspect_ratio <- 1
    #aspect_ratio <- 1.62 * (n_model / n_param)
    pdf(fname, width = 10, height = 10/aspect_ratio,
        pointsize = 15/sqrt(aspect_ratio))
  }
  
  for (i in 1:n_model){
    pred.model <- pred.param[[i]]
    model <- name.model.tex[[name.model[i]]]
    for (j in 1:n_param){
      pred <- pred.model[[j]]
      true <- true.param[[j]]
      plot_pred_vs_true_single(pred, true, name.param[j], param.range.out[[j]], 
                               param.range.in[[j]],
                               main = model, lm = lm, print_mse = print_mse,
                               print_r2 = print_r2)  
    }
  }
  
  if (save){dev.off()}
  
}


getTheilCoefs <- function(pred, true){
  
  n <- length(pred)
  fit <- lm(pred ~ true)
  b1 <- coef(fit)[[2]]
  a <- coef(fit)[[1]]
  pred.lm <- a + b1*true
  
  u_d     <- mean(true**2)
  u_bias  <- n*mean((pred-true))**2
  u_slope <- (b1-1)**2 * sum((true - mean(true))**2)
  u_var   <- sum((pred - pred.lm)**2)
  u_bias  <- u_bias / (n*u_d)
  u_slope <- u_slope/ (n*u_d)
  u_var   <- u_var  / (n*u_d)
  ci      <- confidenceInterval(pred, true) / (u_d)

  # SSD_rec <- u_bias + u_slope + u_var
  SSD <- sum((pred-true)**2)/ (n*u_d)
  SSD_rec <- u_bias + u_slope + u_var
  
  theil_coef <- list("SSD" = SSD, "SSD_rec" = SSD_rec, "bias" = u_bias,
                     "slope" = u_slope, "var" = u_var, "ci"   = ci)
  
  return(theil_coef)
}




confidenceInterval <- function(pred, true){
  
  n   <- length(pred)
  ssd <- (pred - true)**2
  sd  <- sd(ssd)
  se  <- sd / sqrt(n)
  ci  <- 1.96*se
  return(ci)
  
}


create_main_plot_pred_vs_true <- function(rate_name, r2_score, mse_score,
                                          r2 = NA, mse = NA){

  main <- paste('$\\', rate_name, '$', sep = '')
  if (r2_score){main <- paste(main, '| $r^2 = $', r2)}
  if (mse_score){main <- paste(main, '| $NRMSE = $', mse)}
  return(TeX(main))
}


get_significant_code <- function(p_value){
  s_code <- ""
  if (p_value < 0.001){s_code <- "***"}
  else if (p_value < 0.01){s_code <- "**"}
  else if (p_value < 0.05){s_code <- "*"}
  return(s_code)
}


getPredsMLE <- function(type, trees){
  
  if      (type == "crbd") {
    n_param    <- 2
    name.param <- c("lambda", "mu")
  }
  else if (type == "expbd"){
    n_param    <- 4
    name.param <- c("a", "b", "c", "mu")
  }
  else if (type == "bisse"){
    n_param <- 6
    name.param <- c("lambda0", "lambda1", "mu0", "mu1", "q01", "q10") 
  }
  
  pred.param <- vector(mode='list', length=n_param)
  names(pred.param) <- name.param

  n_trees <- length(trees)
  
  for (i in 1:n_trees){
    
    tree <- trees[[i]] # get the tree 

    if (type == "crbd"){
      fit.ape <- ape::birthdeath(tree) # fit bd model 
      pred.r  <- fit.ape$para[[2]] # get r prediction 
      pred.epsilon <- fit.ape$para[[1]] # get epsilon prediction
      # compute lambda and mu from r and epsilon 
      pred <- get_lambda_mu_single(c(pred.r, pred.epsilon))
    }
    
    else if (type == "expbd"){
      f.lambda <- function(t, y) y[1]*exp(-y[2]*t) + y[3]
      f.mu     <- function(t,y)  y[1]
      height <- max(node.age(tree)$ages)
      lambda.init <- c(0.3, 0.3, 0.3)
      mu.init     <- c(0.4)
      result <- fit_bd(tree, height, f.lambda, f.mu, lambda.init, mu.init, f=1,
                       cst.mu = TRUE, expo.lamb = TRUE)
      pred <- abs(c(result$lamb_par, result$mu_par))
    }
    
    else if (type == "bisse"){
      lik <- make.bisse(tree, tree$tip.state)
      lik <- constrain(lik, lambda1 ~ 2*lambda0)
      lik <- constrain(lik, q01 ~ q10)
      lik <- constrain(lik, mu0 ~ 0.)
      lik <- constrain(lik, mu1 ~ 0.)
      p <- starting.point.bisse(tree)
      fit <- find.mle(lik, p, method="subplex")
      pred <- c(as.numeric(fit$par.full))
    }
    
    for (j in 1:n_param){
      param           <- pred[j]
      pred.param[[j]] <- c(pred.param[[j]], param)
    } 
    
    # progress(i, n_trees, progress.bar = TRUE, init = (i==1)) # print progression
  }
  return(pred.param)
}


getPredsMleCrbdApe <- function(trees){
  
  # Set up 
  n_param           <- 2
  name.param        <- c("lambda", "mu")
  pred.param        <- vector(mode='list', length=n_param)
  names(pred.param) <- name.param
  n_trees           <- length(trees)
  
  # Loop on phylogenies 
  for (i in 1:n_trees){
    tree         <- trees[[i]]            # select the phylogeny 
    fit.ape      <- ape::birthdeath(tree) # fit birth-death model 
    pred.r       <- fit.ape$para[[2]]     # r: diversification rate 
    pred.epsilon <- fit.ape$para[[1]]     # epsilon: turnover rate
    
    # compute lambda and mu from r and epsilon 
    pred <- get_lambda_mu_single(c(pred.r, pred.epsilon))
    
    for (j in 1:n_param){
      param           <- pred[j]
      pred.param[[j]] <- c(pred.param[[j]], param)
    } 
    progress(i, n_trees, progress.bar = TRUE, init = (i==1)) 
  }
  return(pred.param)
} 


getPredsMleCrbdDiversitree <- function(trees){
  
  # Set up 
  n_param           <- 2
  name.param        <- c("lambda", "mu")
  pred.param        <- vector(mode='list', length=n_param)
  names(pred.param) <- name.param
  n_trees           <- length(trees)
  
  # Loop on phylogenies 
  for (i in 1:n_trees){
    tree            <- trees[[i]]
    lik             <- diversitree::make.bd(tree)
    fit.diversitree <- find.mle(lik, c(.5, .2)) # fit bd model 
    while (fit.diversitree$code == 3){
      lambda.init <- runif(1,0.1,1)
      mu.init     <- runif(1,0.,1)
      fit.diversitree <- find.mle(lik, c(lambda.init, mu.init))
    }
    if (fit.diversitree$code == 3 ){print(i)}
    pred.lambda     <- fit.diversitree$par[[1]] # get lambda prediction 
    pred.mu         <- fit.diversitree$par[[2]] # get mu prediction 
    param           <- c(pred.lambda, pred.mu)
    
    for (j in 1:n_param){
      #param           <- pred[j]
      pred.param[[j]] <- c(pred.param[[j]], param[j])
    } 
    progress(i, n_trees, progress.bar = TRUE, init = (i==1)) 
  }
  return(pred.param)
}



plot_error_barplot_all <- function(pred.param, true.param, param.range.in, name.param,
                                   save = FALSE, fname = NA){
  

  n_param <- length(true.param) # number of parameters 
  n_model <- length(pred.param) # number of inference models 
  
  par(mfrow = c(n_param, 1)) # prepare plot 

  if (save){
    aspect_ratio <- 1.62
    fname <- paste(fname, "pdf", sep = ".")
    pdf(fname, width = 10/n_param, height = 10/aspect_ratio, pointsize = 15/sqrt(aspect_ratio))
  }
  
  name.param.list <- names(true.param) # list of parameter names
  name.model.list <- names(pred.param) # list of inference model names 
  theil_coef.list <- vector(mode = "list", length = n_param) 
  names(theil_coef.list) <- name.param.list
  
  # Compute the split error for all inference models and all parameters 
  for (i in 1:n_param){
    #name.param <- name.param.list[i] # name of the parameter 
    true <- true.param[[i]] # true parameters vector
    range.in <- param.range.in[[i]] # range of the inner domain of interest 
    for (j in 1:n_model){
      name.model <- name.model.list[[j]] # name of the inference model 
      pred <- pred.param[[j]][[i]] # pred parameters vector
      pred_true.filter <- filter_preds_boundary(pred, true, range.in[1],
                                                range.in[2])
      pred.in <- pred_true.filter$pred.in # predictions in the inner domain
      true.in <- pred_true.filter$true.in # true values in the inner domain 
      theil_coef <- getTheilCoefs(pred.in, true.in) # get the split error 
      theil_coef.list[[i]][[name.model]] <- theil_coef # save split error 
    }
  }
  

  for (i in 1:n_param){
    plot_error_barplot_single(theil_coef.list[[i]], name.param[i])
  }

  if (save){dev.off()}

}


plot_error_barplot_single <- function(theil_coef_list, name.param){
  
  name.model.tex <- list("mle" = TeX("MLE"), "dnn_ss" = TeX("$DNN_{SS}$"), 
                         "cnn_ltt" = TeX("$CNN_{LTT}$"),
                         "rnn_ltt" = TeX("$RNN_{LTT}$"),
                         "cnn_cblv" = TeX("$CNN_{CBLV}$"),
                         "gnn_phylo" = TeX("$GNN_{Phylo}$"),
                         "cnn-ltt" = TeX("$CNN_{LTT}$"), "dnn-ss" = TeX("$DNN_{SS}$"),
                         "rnn-ltt" = TeX("$RNN_{LTT}$"),
                         "cnn-cblv" = TeX("$CNN_{CBLV}$"),
                         "cnn_cblv_notips" = TeX("$CNN_{CBLV}^{notips}$"),
                         "cnn_cblv_disotips" = TeX("$CNN_{CBLV}^{disotips}$"),
                         "gnn-phylo" = TeX("$GNN_{Phylo}$"))
  
  df.error.decomp <- data.frame(matrix(ncol = length(theil_coef_list), nrow = 3))
  colnames(df.error.decomp) <- names(theil_coef_list)
  rownames(df.error.decomp) <- c("Bias Uni.", "Bias Cons.", "Variance")

  ci_center <- c()
  ci_width  <- c()
  
  for (model in 1:length(theil_coef_list)){
    for (error in 3:5){
      err <- theil_coef_list[[model]][[error]]
      df.error.decomp[error - 2, model] <- err
    }
    center <- theil_coef_list[[model]][[1]]
    width  <- theil_coef_list[[model]][[6]]
    ci_center <- c(ci_center, center)
    ci_width  <- c(ci_width , width)
  }
  
  names.arg <- c()
  for (name in colnames(df.error.decomp)){
    names.arg <- c(names.arg, name.model.tex[[name]])
  } 
  
  print(df.error.decomp)

  legend <- c("Bias Uni.", "Bias Cons.", "Variance")

  #print(theil_coef_list)
  
  ylab <- TeX(paste("$\\", name.param, "$ error", sep=""))
  
  ze_barplot <- barplot(as.matrix(df.error.decomp[legend,]), col = c("black", "darkgray","white"),
          ylab = ylab, names.arg = names.arg)
  legend("topleft",                                   
         legend = legend,
         fill = c("black", "darkgray","white"))
  
  error.bar(ze_barplot, ci_center, ci_width)
  
}


#A function to add arrows on the chart
error.bar <- function(x, y, upper, lower=upper, length=0.05,...){
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}


plot_bars_mle_vs_nn <- function(pred.list, true.list, nn_type, name.list, save = FALSE,
                                n_trees = NA, n_taxa = NA, lambda_range = NA, epsilon_range = NA,
                                n_test = NA, n_layer = NA, n_hidden = NA, n_train = NA, ker_size = NA){
  
  n <- length(pred.list[["mle"]])
  print(n)
  par(mfrow=c(1,1))
  
  if (save){
    fname <- get_plot_save_name(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                n_test, n_layer, n_hidden, n_train, ker_size)
    fname <- substring(fname, 1, nchar(fname) - 4)
    fname <- paste(fname, "-bar", ".pdf", sep = "")
    aspect_ratio <- 1.62
    pdf(fname, width = 10, height = 10/aspect_ratio, pointsize = 15/sqrt(aspect_ratio))
  }
  
  
  df.error.decomp <- data.frame(matrix(ncol = n*2, nrow = 3))
  col_names <- c()
  for (name in name.list){
    for (type in c(nn_type, "mle")){
      col_names <- c(col_names, paste(name, type, sep="."))
    }
  }
  colnames(df.error.decomp) <- col_names
  

  for (type in c(nn_type, "mle")){
    preds <- pred.list[[type]]
    for (i in 1:n){
      col  <- paste(name.list[[i]], type, sep=".") 
      pred <- preds[[i]]
      true <- true.list[[i]]
      theil.coef <- get_theil_coefficients(pred, true)
      err.dec <- c(theil.coef$bias, theil.coef$slope, theil.coef$var)
      df.error.decomp[col] <- err.dec
    }
  }
  
  rownames(df.error.decomp) <- c("Bias", "Slope", "Variance")
  barplot(as.matrix(df.error.decomp), col = c("purple", "blue","darkorange"), ylab = "Error")
  legend("topleft",                                   
         legend = c("Bias", "Slope", "Variance"),
         fill = c("purple", "blue","darkorange"))
  
  
  if (save){dev.off()}
  
}



#' Plot the predictions of rates given by MLE
#'
#' Given a list of trees and their true rates, plot the predictions vs. truth
#' where the predictions are obtained by the MLE method.
#'
#' @param trees list of trees 
#' @param vec.true.lambda vector containing the true speciation rates
#' @param vec.true.mu vector containg the true extinction rates 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#' @param type which package to use for the MLE predictions 
#'             either {"ape", "diversitree", "rpanda", "all"}
#'             if "all" use the three packages (ape, diversitree, rpanda) to 
#'             do the predictions
#'             else use only the given package 
#'             (default = "all")
#' @param save logical, should the plot be saved? (default = FALSE)
#' @param fname if save, name of saved file (default = NA)
#' @param alone logical, are the MLE predictions plotted alone or not?
#'              
#' @return 
#' 
#' @export
#' @examples
plot_mle_predictions <- function(trees, vec.true.lambda, vec.true.mu, 
                                 lambda_range, epsilon_range, type = "all", 
                                 save = FALSE, alone = TRUE, bar = TRUE){
  
  n_trees <- length(trees)
  #n_taxa  <- length(trees[[1]]$tip.label)
  
  vec.pred.rpanda.lambda      <- c()
  vec.pred.rpanda.mu          <- c()
  vec.pred.diversitree.lambda <- c()
  vec.pred.diversitree.mu     <- c()
  vec.pred.ape.lambda         <- c()
  vec.pred.ape.mu             <- c()
  
  
  cat("Compute MLE predictions...\n")
  
  for (i in 1:n_trees){
    
    tree <- trees[[i]] # get the phylogenetic tree 
    n_taxa  <- length(tree$tip.label)
    
    # Prepare the fit 
    height <- max(node.age(tree)$ages) # height of the tree 
    f.lambda <-function(t,y){y[1]} # lambda is a function constant over time 
    f.mu     <-function(t,y){y[1]} # mu is a function constant over time 
    lambda_par <- c(.5)  # initial value for lambda
    mu_par     <- c(.4)  # initial value for mu
    
    # RPANDA - Infer lambda & mu 
    if (type == "all" | type == "rpanda"){
      fit.rpanda <- RPANDA::fit_bd(tree, height, f.lambda, f.mu, lambda_par, mu_par, 
                                   cst.lamb = TRUE, cst.mu = TRUE) # fit bd model 
      pred.rpanda.lambda <- fit.rpanda$lamb_par # get lambda prediction 
      pred.rpanda.mu     <- abs(fit.rpanda$mu_par) # get mu prediction 
      vec.pred.rpanda.lambda <- c(vec.pred.rpanda.lambda,
                                  pred.rpanda.lambda) # store lambda prediction
      vec.pred.rpanda.mu     <- c(vec.pred.rpanda.mu,
                                  pred.rpanda.mu) # store mu prediction 
    }
    
    # diversitree - Infer lambda & mu
    if (type == "all" | type == "diversitree"){
      lik  <- diversitree::make.bd(tree)
      fit.diversitree <- diversitree::find.mle(lik, c(lambda_par, mu_par)) # fit bd model 
      pred.diversitree.lambda <- fit.diversitree$par[[1]] # get lambda prediction 
      pred.diversitree.mu     <- fit.diversitree$par[[2]] # get mu prediction 
      vec.pred.diversitree.lambda <- c(vec.pred.diversitree.lambda,
                                       pred.diversitree.lambda) # store lambda prediction
      vec.pred.diversitree.mu     <- c(vec.pred.diversitree.mu,
                                       pred.diversitree.mu) # store mu prediction
    }
    
    # ape - Infer r & epsilon then compute lambda & mu 
    if (type == "all" | type == "ape"){
      fit.ape <- ape::birthdeath(tree) # fit bd model 
      pred.ape.r <- fit.ape$para[[2]] # get r prediction 
      pred.ape.epsilon <- fit.ape$para[[1]] # get epsilon prediction
      # compute lambda and mu from r and epsilon 
      pred.ape.lambda_mu <- get_lambda_mu_single(c(pred.ape.r, pred.ape.epsilon))
      pred.ape.lambda <- pred.ape.lambda_mu[1] # get lambda value 
      pred.ape.mu     <- pred.ape.lambda_mu[2] # get mu value 
      vec.pred.ape.lambda <- c(vec.pred.ape.lambda, pred.ape.lambda) # store lambda prediction
      vec.pred.ape.mu <- c(vec.pred.ape.mu, pred.ape.mu) # store mu prediction 
    }
    
    progress(i, n_trees, progress.bar = TRUE, init = (i==1)) # print progression
    
  }
  
  cat("\nCompute MLE predictions... Done.")
  
  names <- list("lambda", "mu")
  true.list <- list(vec.true.lambda, vec.true.mu)
  
  
  if (type == "all"){subtitles <- list("rpanda", "diversitree", "ape")}
  else{subtitles <- list(paste("MLE:", type), paste("MLE:", type))}
  
  if (type == "rpanda"){
    pred.list <- list(vec.pred.rpanda.lambda, vec.pred.rpanda.mu)
  }
  
  else if (type == "diversitree"){
    pred.list <- list(vec.pred.diversitree.lambda, vec.pred.diversitree.mu)
  }
  
  else if (type == "ape"){
    pred.list <- list(vec.pred.ape.lambda, vec.pred.ape.mu)
  }
  
  else if (type == "all"){
    pred.list <- list(vec.pred.rpanda.lambda, vec.pred.rpanda.mu,
                      vec.pred.diversitree.lambda, vec.pred.diversitree.mu, 
                      vec.pred.ape.lambda, vec.pred.ape.mu)
  }
  
  else{print("Error: type unkown. Type should be either: 'rpanda', 'diversitree', 'ape' or 'all'.")}
  
  
  if (save){
    fname <- get_plot_save_name("mle", n_trees, n_taxa, lambda_range, epsilon_range,
                                n_test)
  }
  
  plot_pred_vs_true(pred.list, true.list, names, save = save, fname = fname, 
                    subtitle = subtitles, alone = alone, bar = bar)
  
}


#' Plot together predictions from Neural Networks and MLE 
#' 
#'
#' @param pred.nn.list list of predicted parameters by the Neural Network
#' @param true.list true values of the parameters 
#' @param trees list of trees on which the predictions of rates have been made
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#' @param mle_package which MLE package to use (either "ape", "diversitree", "rpanda")
#'                    (default = "ape")
#' @param nn_type type of Neural Network (e.g. CNN, DNN, RNN) (default = "NN")
#' @param save logical, should the plot be saved? (default = FALSE)
#' @param fname if save, file name
#' 
#'              
#' @return fname, file name of the prediction pots 
#' 
#' @export
#' @examples
plot_together_nn_mle_predictions <- function(pred.nn.list, true.list, trees, nn_type,
                                             n_trees, n_taxa, lambda_range, epsilon_range,
                                             n_layer, n_hidden, n_train, ker_size = NA,
                                             mle_package = "ape", 
                                             save = FALSE, bar = FALSE){
  
  
  n_row <- 2
  n_col <- 2 + as.integer(bar)
  
  if (save){
    n_test <- length(trees)
    fname <- get_plot_save_name(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                n_test, n_layer, n_hidden, n_train, ker_size)
    aspect_ratio <- 1.62 * n_col/n_row
    pdf(fname, width = 10, height = 10/aspect_ratio, pointsize = 15/sqrt(aspect_ratio))
  }
  
  name.list <- list("lambda", "mu")
  
  par(mfrow=c(n_row,n_col))
  plot_pred_vs_true(pred.nn.list, true.list, name.list,
                    subtitles = c(nn_type, nn_type), save = FALSE,
                    alone = FALSE, bar = bar)
  vec.true.lambda <- true.list[[1]]
  vec.true.mu     <- true.list[[2]]
  plot_mle_predictions(trees, vec.true.lambda, vec.true.mu, 
                       lambda_range, epsilon_range, bar = bar,
                       type = mle_package, save = FALSE, alone = FALSE)
  
  if (save){dev.off()}
  
}


#### end ####


#### Generating, Converting, Saving and Loading ####


drawPhyloSize <- function(n_taxa){
  size <- ifelse(length(n_taxa) == 2, sample(n_taxa[1]:n_taxa[2], 1), n_taxa)
  return(size)
}


drawRateCRBD <- function(param.range){
  
  n_param <- length(param.range)
  if (n_param != 2){print("Wong number of parameters. CRBD has 2 parameters.")}
  # Param 1 = lambda  = speciation rate 
  # Param 2 = epsilon = turnover   rate  
  
  vec.param <- c()
  for (i in 1:n_param){
    range <- param.range[[i]]               # parameter range
    param <- runif(1, range[1], range[2])   # draw randomly the parameter
    vec.param <- c(vec.param, param)        # save
  }
  
  # Transform epsilon into mu (extinction rate)
  vec.param[2] <- vec.param[1]*vec.param[2] # mu = epsilon * lambda
  
  return(vec.param)
}


drawRateBiSSE <- function(param.range){
  
  lambda.range <- param.range[[1]]
  q.range      <- param.range[[2]]
  lambda0 <- runif(1, lambda.range[1], lambda.range[2])
  lambda1 <- 2*lambda0
  mu0     <- 0.
  mu1     <- 0.
  q01     <- runif(1, q.range[1], q.range[2])
  q10     <- q01
  vec.param <- c(lambda0, lambda1, mu0, mu1, q01, q10)
  return(vec.param)
  
}


#' Generate phylogenetic trees 
#'
#' Create a given number of trees of a choosen size and rates. This trees are
#' stored in a list.
#'
#' @param n_trees number of trees to generate 
#' @param n_taxa size of the trees 
#' @param lambda_range vector containg the min and max of the speciation rates 
#'                     of the trees generated. For each trees the speciation 
#'                     rate will randomly drawn in this interval
#' @param lambda_range vector containg the min and max of the turnover rates 
#'                     of the trees generated. For each trees the turnover 
#'                     rate will randomly drawn in this interval
#' @param ss_check logical, should we check that the summary statistics 
#'                 corresponding to the generated trees doesn't contain any NA
#'                 values (default = TRUE)
#'              
#' @return out list of outputs 
#'         $trees -> trees, the list of the generated trees 
#'         $lambda -> vec.lambda, vector of speciation rates of the generated trees
#'         $mu -> vec.mu, vector of extinction rates of generated trees
#' 
#' @export
#' @examples
generatePhyloCRBD <- function(n_trees, n_taxa, param.range, ss_check = TRUE){
  
  # Initialization
  trees      <- list() # where trees will be stored 
  n_param    <- length(param.range) # number of parameters
  true.param <- vector(mode='list', length=n_param)
  names(true.param) <- c("lambda", "mu") 
  
  cat("Generation of phylogenies...\n")
  
  while (length(trees) < n_trees){
    
    # Generate the phylogenetic tree parameters 
    vec.param <- drawRateCRBD(param.range)
    
    # Draw the number of tips of the tree
    n_taxa.i <- drawPhyloSize(n_taxa)
    
    # Generate tree
    tree <- trees(c(vec.param[1], vec.param[2]), "bd", max.taxa=n_taxa.i)[[1]] 
    
    # Checking that Summary Statistics have no NA
    if (ss_check){
      ss       <- get_ss(tree)    # compute summary statistics
      no_NA_ss <- !any(is.na(ss)) # does SS have any NA values?
    }
    
    # If no checking or SS without NAs
    if (no_NA_ss || !ss_check){
      trees <- append(trees, list(tree)) # save tree
      for (i in 1:n_param){
        true.param[[i]] <- c(true.param[[i]], vec.param[i]) # save param.
      }
      progress(length(trees), n_trees, progress.bar = TRUE, # print
               init = (length(trees)==1))                   # progression
    }
  }
  
  cat("\nGeneration of phylogenies... Done.")
  
  # Prepare output containing: trees (list), true param values (vector).
  out <- list("trees"    = trees, 
              "param"    = true.param)
  
  return(out)
}




generatePhyloBiSSE <- function(n_trees, n_taxa, param.range, ss_check = TRUE){
  
  trees <- list()
  name.param <- c("lambda0", "lambda1", "mu0", "mu1", "q01", "q10") 
  true.param <- vector(mode='list', length=6)
  names(true.param) <- name.param
  
  while(length(trees) < n_trees){
    vec.param <- drawRateBiSSE(param.range) # draw randomly param. values
    n_taxa.i <- drawPhyloSize(n_taxa)       # draw phylogeny size 
    
    # Generate phylogeny 
    tree <- NULL
    lik  <- NULL
    while(is.null(tree) | is.null(lik)){
      tree <- tree.bisse(vec.param, max.taxa = n_taxa.i, x0 = NA)
      if (!(all(tree$tip.state == 0) | all(tree$tip.state == 1))){
        lik <- make.bisse(tree, tree$tip.state)
      }
    }
    
    # Checking that summary statistics have no NA
    if (ss_check){
      ss <- get_ss(tree) # compute summary statistics
      no_NA_ss <- !any(is.na(ss)) # does SS have any NA values?
    }
    
    if (no_NA_ss || !ss_check){
        trees <- append(trees, list(tree))                    # save tree
        for (i in 1:6){
          true.param[[i]] <- c(true.param[[i]], vec.param[i]) # save param.
        }
        progress(length(trees), n_trees, progress.bar = TRUE, # print
                 init = (length(trees)==1))                   # progression
      }
    }
  
  out <- list("trees"    = trees, 
              "param"    = true.param)
  
  return(out)
}


generatePhylo <- function(model, n_trees, n_taxa, param.range, ss_check = TRUE){
  
  if (model == "crbd"){
    out <- generatePhyloCRBD(n_trees, n_taxa, param.range, ss_check)
  } else if (model == "bisse"){
    out <- generatePhyloBiSSE(n_trees, n_taxa, param.range, ss_check)
  } else {print("Model unkown, model should either crbd or bisse.")}

}


#' Generate a data.frame containing Lineage Through Time of phylo trees
#'
#' Given a list of phylo trees, returns their LTT in a data.frame
#'
#' @param trees list of phylo trees
#' @param n_taxa size of the trees 
#' @param vec.lambda vector of the corresponding speciation rates of the
#'                   generated trees 
#' @param vec.mu vector of the corresponding extinction rates of the
#'                   generated trees                   
#'              
#' @return out list of outputs
#'         $ltt -> df.ltt data.frame containing the LTT of the trees 
#'                i'th column contains the LTT of the i'th tree
#'         $rates -> df.rates data.frame containing the speciation and extinction rates  
#' 
#' @export
#' @examples
generate_ltt_dataframe <- function(trees, n_taxa, true.param){
  
  n_trees  <- length(trees) # number of trees 
  n_row <- ifelse(length(n_taxa) == 1, n_taxa, n_taxa[2])
  df.ltt <- data.frame("tree1" = rep(NA, n_row))
  
  df.rates <- as.data.frame(do.call(cbind, true.param))
  
  cat("Creating LTT dataframe...\n")
  
  for (i in 1:n_trees){
    tree <- trees[[i]] # get tree 
    ltt.coord <- ape::ltt.plot.coords(tree) # get ltt coordinates 
    ltt.coord <- as.data.frame(ltt.coord)
    ltt.coord.time <- ltt.coord$time
    n <- length(ltt.coord.time)
    df.ltt[1:n,paste("tree", i, sep = "")] <- ltt.coord$time
    progress(i, n_trees, progress.bar = TRUE, init = (i==1))
  }
  
  cat("\nCreating LTT dataframe... Done.")

  out <- list("ltt" = df.ltt, "rates" = df.rates) # function output
  
  return(out)
  
}

generate_ltt_dataframe2 <- function(trees, n_taxa){
  
  n_trees  <- length(trees) # number of trees 
  n_row <- ifelse(length(n_taxa) == 1, n_taxa, n_taxa[2])
  df.ltt <- data.frame("tree1" = rep(NA, n_row))
  
  cat("Creating LTT dataframe...\n")
  
  for (i in 1:n_trees){
    tree <- trees[[i]] # get tree 
    ltt.coord <- ape::ltt.plot.coords(tree) # get ltt coordinates 
    ltt.coord <- as.data.frame(ltt.coord)
    ltt.coord.time <- ltt.coord$time
    n <- length(ltt.coord.time)
    df.ltt[1:n,paste("tree", i, sep = "")] <- ltt.coord$time
    progress(i, n_trees, progress.bar = TRUE, init = (i==1))
  }
  
  cat("\nCreating LTT dataframe... Done.")

  out <- list("ltt" = df.ltt) # function output
  
  return(out)
  
}


#' Convert a LTT data.frame into a torch dataset 
#'
#' Given LTT data.frame and a data.frame containing rates (targets to infer)
#' create the corresponding torch dataset 
#'
#' @param df.ltt data.frame containing LTT (created by generate_ltt_dataframe)
#' @param df.rates data.frame containing rates (created by generated_ltt_dataframe)
#' @param vec.lambda vector of the corresponding speciation rates of the
#'              
#' @return torch dataset 
#'         $x -> LTT contained in df.ltt
#'         $y -> corresponding rates contained in df.rates (targets)
#' 
#' @export
#' @examples
convert_ltt_dataframe_to_dataset <- function(df.ltt, true.param, nn_type){
  
  if (nn_type == "cnn-ltt"){
    ds.ltt <- torch::dataset(
      name <- "ltt_dataset", 
      initialize = function(df.ltt, true.param){
        
        # input
        df.ltt[is.na(df.ltt)] <- 0
        
        array.ltt <- df.ltt %>% 
          as.matrix() %>% 
          torch_tensor()
        self$x <- array.ltt
        
        # target 
        self$y <- torch_tensor(do.call(cbind, true.param)) # target
      }, 
      
      .getitem = function(i) {list(x = self$x[,i]$unsqueeze(1), y = self$y[i, ])},
      
      .length = function() {self$y$size()[[1]]}
  )
  }
  
  else{
    ds.ltt <- torch::dataset(
      name <- "ltt_dataset", 
      initialize = function(df.ltt, true.param){
        
        # input
        df.ltt[is.na(df.ltt)] <- 0
        
        array.ltt <- df.ltt %>% 
          as.matrix() %>% 
          torch_tensor()
        self$x <- array.ltt
        
        # target 
        self$y <- torch_tensor(do.call(cbind, true.param)) # target
      }, 
      
      .getitem = function(i) {list(x = self$x[,i], y = self$y[i, ])},
      
      .length = function() {self$y$size()[[1]]}
    )
  }
  
  return(ds.ltt)
}


convert_ltt_dataframe_to_dataset_sizes <- function(df.ltt, df.rates){
  
  ds.ltt <- torch::dataset(
    name <- "ltt_dataset", 
    initialize = function(df.ltt, df.rates){
      
      # input data
      # df.ltt[is.na(df.ltt)] <- 0
      
      #array.ltt <- df.ltt %>% 
      #  as.matrix() %>% 
      #  torch_tensor()
      
      # input data 
      x <- df.ltt
      self$x <- x
      
      # target data 
      target.names <- c("lambda", "mu")
      y <- df.rates[target.names] %>% 
        as.matrix()
      self$y <- torch_tensor(y)
      
    }, 
    
    .getitem = function(i) {
      x <- df.ltt[i] %>% na.omit() %>% as.matrix() %>% torch_tensor()
      #x <- as.numeric(na.omit(self$x[,i]))
      list(x = x, y = self$y[i, ])
    }, 
    
    .length = function() {
      self$y$size()[[1]]
    }
  )
  
  return(ds.ltt)
}


convert_encode_to_dataset <- function(tensor.encode, true.param){
  
  ds.encode <- torch::dataset(
    
    initialize = function(tensor.encode, true.param){
      self$x <- torch_tensor(tensor.encode) # input 
      self$y <- torch_tensor(do.call(cbind, true.param)) # target
    },
    .getitem = function(i) {list(x = self$x[,i]$unsqueeze(1), y = self$y[i,])},
    .length = function() {self$y$size()[[1]]}
  )
 return(ds.encode)
}


# convert_encode_to_dataset <- function(tensor.encode, true.param) {
#   ds.encode <- torch::dataset(
#     initialize = function(tensor.encode, true.param) {
#       self$x <- torch_tensor(tensor.encode) # input 
#       self$y <- torch_tensor(do.call(cbind, true.param)) # target
#     },
#     .getitem = function(i) {
#       list(x = self$x[i, ]$unsqueeze(1), y = self$y[i, ])
#     },
#     .length = function() {
#       self$y$size()[[1]]
#     }
#   )
#   return(ds.encode)
# }

# extract_dataset <- function(dataset) {
#   length <- dataset$.length()
#   data_list <- lapply(1:length, function(i) {
#     item <- dataset$.getitem(i)
#     list(x = as.numeric(item$x), y = as.numeric(item$y))
#   })
#   df <- do.call(rbind, data_list)
#   return(as.data.frame(df))
# }

filter_preds_boundary <- function(pred, true, lower = .15, upper = .8){
  
  n <- length(true)
  indices.in  <- which(true > lower & true < upper)
  indices.out <- setdiff(1:n, indices.in)
  out <- list("pred.in"  = pred[indices.in], 
              "true.in"  = true[indices.in],
              "pred.out" = pred[indices.out], 
              "true.out" = true[indices.out])
  
  return(out)
}



get_backbone_save_name <- function(n_trees, n_taxa, param.range){
  
  n_taxa_text <- ifelse(length(n_taxa) == 2, paste(n_taxa[1], n_taxa[2], sep="-"), 
                        as.character(n_taxa))
  
  fname <- paste("ntrees", n_trees, "ntaxa", n_taxa_text, sep="-")
  
  for (i in 1:length(param.range)){
    fname <- paste(fname, names(param.range)[i], param.range[[i]][1],
                   param.range[[i]][2], sep = "-")
  }
  
  return(fname)
}


get_model_save_name <- function(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                 n_layer, n_hidden, n_train, ker_size = NA){
  
  fname <- get_backbone_save_name(n_trees, n_taxa, lambda_range, epsilon_range)
  fname.model <- paste(fname, "nlayer", n_layer, "nhidden", n_hidden, 
                       "ntrain", n_train, sep = "-")
  if (!is.na(ker_size)){fname.model <- paste(fname.model, "kersize", ker_size, sep = "-")}
  dir <- paste("neural-networks-models", nn_type, "", sep = "/")
  fname.model <- paste(dir, fname.model, sep = "")
  return(fname.model)
}


getSaveName <- function(n_trees, n_taxa, param.range, ss_check = TRUE){
  
  dir <- "trees-dataset/"
  
  fname <- get_backbone_save_name(n_trees, n_taxa, param.range)
  fname <- paste(fname, "sscheck", ss_check, sep = "-")
  
  fname.trees  <- paste(dir, fname, "-trees.rds", sep="")
  fname.param  <- paste(dir, fname, "-param.rds", sep="")
  fname.ss     <- paste(dir, fname, "-ss.rds", sep="")
  fname.mle    <- paste(dir, fname, "-mle.rds", sep="")
  fname.encode <- paste(dir, fname, "-encode.rds", sep="")
  fname.ltt    <- paste(dir, fname, "-ltt.rds", sep="")
  fname.df     <- paste(dir, fname, "-df.rds", sep="")
  
  fnames <- list("trees"  = fname.trees, 
                 "param"  = fname.param, 
                 "ss"     = fname.ss, 
                 "mle"    = fname.mle, 
                 "encode" = fname.encode, 
                 "ltt"    = fname.ltt,
                 "df"     = fname.df)
  
  return(fnames)
  
}


#' Save generated trees and their rates 
#'
#' Given a list of trees and their rates, those are saved using saveRDS function
#'
#' @param trees list of trees 
#' @param vec.lambda vector containing the speciation rates
#' @param vec.mu vector containg the extinction rates 
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#' @param ss_check logical, have we check that the trees generated doesn't 
#'                 give NA values for the summary statistics
#'              
#' @return 
#' 
#' @export
#' @examples
savePhylogeny <- function(trees, true.param, n_trees, n_taxa,
                          param.range, ss_check = TRUE){
  
  # Getting file names to save 
  fnames <- get_dataset_save_name(n_trees, n_taxa, param.range, ss_check)
  fname.trees  <- fnames$trees
  fname.param  <- fnames$param

  # Saving data 
  cat("Saving data...\n")
  saveRDS(trees, fname.trees)
  cat(paste(fname.trees, " saved.\n", sep=""))
  saveRDS(true.param, fname.param)
  cat(paste(fname.param, " saved.\n", sep=""))
  cat("Saving data... Done.\n")
  
}


#' Load trees list and rates vector
#'
#' Load a saved list of trees and corresponding rates, 
#' given the information needed to recover their names
#'
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#' @param ss_check ss_check logical, have we check that the trees generated doesn't 
#'                 give NA values for the summary statistics
#'              
#' @return out list of outputs 
#'         $trees -> trees, the list of the generated trees 
#'         $lambda -> vec.lambda, vector of speciation rates of the generated trees
#'         $mu -> vec.mu, vector of extinction rates of generated trees
#' 
#' @export
#' @examples
readPhylogeny <- function(n_trees, n_taxa, param.range, ss_check = TRUE,
                          load_trees = TRUE){
  
  # Getting file names to load 
  fnames <- getSaveName(n_trees, n_taxa, param.range, ss_check)
  fname.trees  <- fnames$trees
  fname.param  <- fnames$param

  # Loading data 
  cat("Loading data...\n")
  if (load_trees){
    trees <- readRDS(fname.trees)
    cat(paste(fname.trees, " loaded.\n", sep=""))}
  print(fname.param)
  param <- readRDS(fname.param)
  cat(paste(fname.param, " loaded.\n", sep=""))
  cat("Loading data... Done.\n")
  
  out <- list("param" = param)
  if (load_trees){out$trees <- trees}
  
  return(out)
  
}


#' Save computed trees summary statistics data.frame
#'
#' 
#'
#' @param df.ss data.frame containing the summary statics and rates of several trees
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#'              
#' @return 
#' 
#' @export
#' @examples
saveSummaryStatistics <- function(df.ss, n_trees, n_taxa, param.range){
  
  fnames   <- get_dataset_save_name(n_trees, n_taxa, param.range, ss_check = TRUE)
  fname.ss <- fnames$ss
  
  cat("Saving summary statistics data...\n")
  saveRDS(df.ss, fname.ss)
  cat(paste(fname.ss, " saved.\n", sep=""))
  cat("Saving summary statistics data... Done.\n")
  
}


#' Load a saved summary statistics data.frame
#'
#' Load a summary statistics data.frame, given the information needed to recover
#' its name 
#'
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#'              
#' @return df.ss summary statistics data.frame
#' 
#' @export
#' @examples
readSummaryStatistics <- function(n_trees, n_taxa, param.range){
  
  fnames <- getSaveName(n_trees, n_taxa, param.range, ss_check = TRUE)
  fname.ss <- fnames$ss
  
  cat("Loading summary statistics data...\n")
  df.ss <- readRDS(fname.ss)
  cat("Loading summary statistics data... Done.\n")
  
  return(df.ss)
  
}


save_predictions <- function(pred.list, true.list, nn_type, n_trees, n_taxa,
                             lambda_range, epsilon_range, n_test, n_layer, 
                             n_hidden, n_train, ker_size = NA){
  
  cat("Saving predictions...\n")
  save.list <- list("pred" = pred.list, "true" = true.list)
  fname.preds <- get_preds_save_name(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                          n_test, n_layer, n_hidden, n_train, ker_size)
  saveRDS(save.list, fname.preds)
  cat("\nSaving predictions... Done.\n")
  
}


load_predictions <- function(nn_type, n_trees, n_taxa,
                             lambda_range, epsilon_range, n_test, n_layer, 
                             n_hidden, n_train, ker_size = NA){
  
  fname.preds <- get_preds_save_name(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                     n_test, n_layer, n_hidden, n_train, ker_size)
  
  save.list <- readRDS(fname.preds)
  
  return(save.list)
  
}


#' Create file name for the prediction plots 
#'
#'
#' @param n_trees number of trees in trees 
#' @param n_taxa size of the trees 
#' @param lambda_range range in which the speciation rates have been generated
#' @param epsilon_range range in which the turnover rates have been generated 
#' @param n_test number of tested predictions
#' @param dir directory where the file will be saved 
#' @param type if the model is a neural network, type should be "nn" such that,
#'             relevant informations will be added to the file name
#' @param n_layer if type="nn", number of layer of the neural network (default = NA)
#' @param n_hidden if type="nn", number of neurons in hidden layeres (default = NA)
#' @param n_train if type="nn", size of the training set (default = NA)
#' 
#'              
#' @return fname, file name of the prediction pots 
#' 
#' @export
#' @examples
get_plot_save_name <- function(model_type, n_trees, n_taxa, lambda_range, epsilon_range,
                               n_test, n_layer = NA, n_hidden = NA, n_train = NA, ker_size = NA){
  
  fname <- get_backbone_save_name(n_trees, n_taxa, lambda_range, epsilon_range)

  if (model_type != "mle") {
    fname.plot <- paste(fname, "nlayer", n_layer, "nhidden", n_hidden, 
                       "ntrain", n_train, sep = "-")
    if (!is.na(ker_size)){fname.plot <- paste(fname.plot, "kersize", ker_size, sep = "-")}
  }

  dir.plot <- paste("figures", model_type, "", sep = "/")
  fname.plot <- paste(dir.plot, fname.plot, "-ntest-", n_test, sep = "")
  fname.plot <- paste(fname.plot, "pdf", sep = ".")
  
  return(fname.plot)
  
}


getSaveNamePredsMLE <- function(n_trees, n_taxa, param.range, ss_check =TRUE){
  fname <- get_dataset_save_name(n_trees, n_taxa, param.range, ss_check)$ss
  fname <- substring(fname, 1, nchar(fname) - 6)
  fname <- paste(fname, "mle.rds", sep = "")
  return(fname)
}


savePredsMLE <- function(pred.param, n_trees, n_taxa, param.range,
                         ss_check = TRUE){
  fname <- getSaveNamePredsMLE(n_trees, n_taxa, param.range, ss_check)
  saveRDS(pred.param, fname)
}



get_preds_save_name <- function(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                 n_test, n_layer, n_hidden, n_train, ker_size = NA){
  
  fname.plot <- get_plot_save_name(nn_type, n_trees, n_taxa, lambda_range, epsilon_range,
                                   n_test, n_layer, n_hidden, n_train, ker_size)
  
  start <- nchar("figures") + nchar(nn_type) + 3
  fname.preds <- substring(fname.plot, start, nchar(fname.plot) - 4)
  dir.preds   <- paste("neural-networks-predictions", nn_type, "", sep = "/")
  fname.preds <- paste(dir.preds, fname.preds, ".rds", sep = "")
  
  return(fname.preds)
}


df_add_tipstate <- function(df, trees){
  
  n_trees <- length(trees)
  
  vec_prop.tip.state <- c()
  
  for (n in 1:n_trees){
    tree <- trees[[n]]
    prop.tip.state <- mean(as.numeric(tree$tip.state))
    vec_prop.tip.state <- c(vec_prop.tip.state, prop.tip.state)
  }
  
  df$prop.tip.state <- vec_prop.tip.state
  
  return(df)
}

extract_elements <- function(list_of_vectors, indices_to_extract){
  l <- as.list(do.call(cbind, list_of_vectors)[indices_to_extract,] 
               %>% as.data.frame())
  return(l)
}


#### end #### 


#### RNN custom batch ####


find_indices_same_size <- function(df.ltt, n_taxa){
  list.indices <- list()
  for (n in n_taxa[1]:n_taxa[2]){
    list.indices[[as.character(n)]] <- c()
  }
  for(i in 1:ncol(df.ltt)){
    n <- nrow(na.omit(df.ltt[i]))
    list.indices[[as.character(n)]] <- c(list.indices[[as.character(n)]], i)
  }
  return(list.indices)
}


create_one_batch <- function(df.ltt, true.param, indices){
  
  batch <- list("x" = NA, "y" = NA)
  
  inputs <- na.omit(df.ltt[indices]) %>%
    as.matrix() %>% t() %>%
    torch_tensor()
  
  df.param <- true.param %>% as.data.frame()
  
  targets <- df.param[indices,] %>%
    as.matrix %>%
    torch_tensor()
  
  batch$x <- inputs
  batch$y <- targets
  
  return(batch)
  
}


create_all_batch <- function(df.ltt, true.param, list.indices, n_taxa){
  
  list.batch <- list()
  #n_iter <- sample(n_taxa[1]:n_taxa[2], length(n_taxa[1]:n_taxa[2]))
  n_tot <- length(list.indices)
  for (n in 1:n_tot){
    indices <- list.indices[[n]]
    batch <- create_one_batch(df.ltt, true.param, indices)
    if (dim(batch$x)[1] != 0){
      list.batch[[n]] <- batch
    }
  }
  return(list.batch)
}


reformat_set <- function(set, max_batch_size){
  
  new_set <- list()
  n_batch <- length(set)
  
  for (i in 1:n_batch){
    batch <- set[[i]]
    batch_size <- dim(batch$y)[1]
    n_new_batch <- (batch_size - 1) %/% (max_batch_size) + 1
    for (j in 1:n_new_batch){
      new_batch   <- list()
      start       <- 1 + (j-1)*max_batch_size
      end         <- min(c(batch_size,j*max_batch_size))
      new_batch$x <- batch$x[start:end,]
      new_batch$y <- batch$y[start:end,]
      new_set[[length(new_set)+1]] <- new_batch
    }
  }
  
  return(new_set)
  
}



reformat_test_batch <- function(test.batch){
  
  new.test.batch <- list()
  n_test_batch <- length(test.batch)
  
  for (i in 1:n_test_batch){
    batch <- test.batch[[i]]
    n <- dim(batch$x)[1]
    for (j in 1:n){
      mini.batch <- list("x" = NA, "y" = NA)
      input  <- batch$x[j,]
      target <- batch$y[j,]
      mini.batch$x <- input
      mini.batch$y <- target 
      new.test.batch[[length(new.test.batch) + 1]] <- mini.batch
    }
  }
  
  return(new.test.batch)
}

recover_indices <- function(list.indices){
  n <- length(list.indices)
  recover.ind <- c()
  for (i in 1:n){
    ind <- list.indices[[i]]
    recover.ind <- c(recover.ind, ind)
  }
  
  return(recover.ind)
}


convert_pred_GNN <- function(pred_GNN_from_py){
  
  n <- length(pred_GNN_from_py)
  pred_GNN <- list()
  
  for (i in 1:n){
    pred <- c()
    m <- length(pred_GNN_from_py[[i]])
    for (j in 1:m){
      pred <- c(pred, pred_GNN_from_py[[i]][[j]])
    }
    pred_GNN[[i]] <- pred
  }
  
  return(pred_GNN)
  
}

#### end ####






