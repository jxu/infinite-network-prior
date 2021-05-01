library(tidyverse)

gaussian_matrix <- function(nrow, ncol, mean, sd) {
  matrix(rnorm(nrow*ncol, mean, sd), nrow, ncol)
}

# simple random weights one hidden layer NN, described in Neal (1996)
one_layer_nn <- function(x, hidden_dim, output_dim) {
  set.seed(10716)
  input_dim <- length(x)
  
  # u, a: hidden layer weights, bias
  # v, b: output layer weights, bias
  
  u <- gaussian_matrix(hidden_dim, input_dim, 0, 1)  
  a <- rnorm(hidden_dim, 0, 1)  
  v <- gaussian_matrix(output_dim, hidden_dim, 0, hidden_dim^(-1/2)) 
  b <- rnorm(output_dim, 0, 1) 
  
  h <- tanh(a + u %*% x)
  y <- b + v %*% h
  
  return(y)
  
}

one_layer_nn(1, 2, 1)


