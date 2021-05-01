library(tidyverse)

set.seed(10716)

gaussian_matrix <- function(nrow, ncol, mean, sd) {
  matrix(rnorm(nrow*ncol, mean, sd), nrow, ncol)
}

uniform_matrix <- function(nrow, ncol, min, max) {
  matrix(runif(nrow*ncol, min, max), nrow, ncol)
}

# simple random weights one hidden layer NN, described in Neal (1996)
one_layer_nn <- function(x, hidden_dim, output_dim) {
  input_dim <- length(x)
  
  # u, a: hidden layer weights, bias
  # v, b: output layer weights, bias
  
  u <- uniform_matrix(hidden_dim, input_dim, -1, 1)  
  a <- rnorm(hidden_dim, 0, 1)  
  v <- gaussian_matrix(output_dim, hidden_dim, 0, hidden_dim^(-1/2)) 
  b <- rnorm(output_dim, 0, 1) 
  
  h <- tanh(a + u %*% x)
  y <- b + v %*% h
  
  return(as.numeric(y))
  
}

# simulate many networks to see output distribution
# uses fixed data model

# 1D output
x <- rep(1,10)
sim_1d_5 <- data.frame(x=replicate(1000, one_layer_nn(x, 5, 1)))
ggplot(sim_1d_5, aes(x = x)) +
  geom_density()

sim_1d_100 <- data.frame(x=replicate(1000, one_layer_nn(x, 100, 1)))
ggplot(sim_1d_100, aes(x = x)) +
  geom_density()


# 2D output
sim_2d_5 <- replicate(10000, one_layer_nn(x, 5, 2))
sim_2d_5 <- data.frame(x=sim_2d_5[1,], y=sim_2d_5[2,])
ggplot(sim_2d_5, aes(x=x, y=y)) +
  geom_density_2d() + 
  coord_cartesian(xlim=c(-3,3), ylim=c(-3,3))


sim_2d_100 <- replicate(10000, one_layer_nn(x, 100, 2))
sim_2d_100 <- data.frame(x=sim_2d_100[1,], y=sim_2d_100[2,])
ggplot(sim_2d_100, aes(x=x, y=y)) +
  geom_density_2d() + 
  coord_cartesian(xlim=c(-3,3), ylim=c(-3,3))


