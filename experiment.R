library(tidyverse)
library(gridExtra)

set.seed(10716)

gaussian_matrix <- function(nrow, ncol, mean, sd) {
  matrix(rnorm(nrow*ncol, mean, sd), nrow, ncol)
}

uniform_matrix <- function(nrow, ncol, min, max) {
  matrix(runif(nrow*ncol, min, max), nrow, ncol)
}

# simple random weights one hidden layer NN, described in Neal (1996)
# use ReLU instead of tanh to make convergence clearer, though convergence holds for both
# low weight for Guassian bias to give more weight to weights, like in Jacot et al (2018)
one_layer_nn <- function(x, hidden_dim, output_dim) {
  input_dim <- length(x)
  
  # u, a: hidden layer weights, bias
  # v, b: output layer weights, bias
  u <- gaussian_matrix(hidden_dim, input_dim, 0, 1)  
  a <- rnorm(hidden_dim, 0, 1)  
  v <- gaussian_matrix(output_dim, hidden_dim, 0, 1) / sqrt(hidden_dim)
  b <- 0.1 * rnorm(output_dim, 0, 1) 
  
  h <- pmax(a + u %*% x, 0)
  y <- b + v %*% h
  
  return(as.numeric(y))
  
}


# simulate many networks to see output distribution
# uses fixed data model

# 1D output
plot_sim_1d <- function(hidden_dim, trials=1000) {
  x <- 1  # fixed data
  samp <- replicate(trials, one_layer_nn(x, hidden_dim, 1))
  ggplot(data.frame(x = samp), aes(x = x)) +
    geom_density() +
    coord_cartesian(xlim=c(-3,3)) +
    stat_function(fun=dnorm, xlim=c(-3,3), args = list(mean=0, sd=1.0), color="blue") +
    ggtitle(paste0("hidden=", hidden_dim))
}


grobs <- lapply(c(1, 5, 10, 1000), plot_sim_1d)
grid.arrange(grobs=grobs)


# 2D output
plot_sim_2d <- function(hidden_dim, trials=1000) {
  x1 <- c(0,1)  # fixed data
  x2 <- c(1,0)
  samp1 <- replicate(trials, one_layer_nn(x1, hidden_dim, 1))
  samp2 <- replicate(trials, one_layer_nn(x2, hidden_dim, 1))
  
  df <- data.frame(x=samp1, y=samp2)
  ggplot(df, aes(x=x, y=y)) +
    stat_density_2d(aes(fill = ..level..), geom = "polygon") +
    coord_cartesian(xlim=c(-3,3), ylim=c(-3,3)) +
    ggtitle(paste0("hidden=", hidden_dim))
  
}

grobs <- lapply(c(1, 5, 10, 1000), plot_sim_2d)
grid.arrange(grobs=grobs)


