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
# low weight for bias like in Jacot et al (2018)
one_layer_nn <- function(x, hidden_dim, output_dim) {
  input_dim <- length(x)
  
  # u, a: hidden layer weights, bias
  # v, b: output layer weights, bias
  
  u <- gaussian_matrix(hidden_dim, input_dim, 0, 1)  
  a <- rnorm(hidden_dim, 0, 1)  
  v <- gaussian_matrix(output_dim, hidden_dim, 0, 1) / sqrt(hidden_dim)
  b <- 0.1 * rnorm(output_dim, 0, 1) 
  
  h <- tanh(a + u %*% x)
  y <- b + v %*% h
  
  return(as.numeric(y))
  
}


# simulate many networks to see output distribution
# uses fixed data model

# 1D output
simulate_1d <- function(hidden_dim, trials=1000) {
  x <- 1  # fixed input
  samp <- replicate(trials, one_layer_nn(x, hidden_dim, 1))
  ggplot(data.frame(x = samp), aes(x = x)) +
    geom_density() +
    coord_cartesian(xlim=c(-3,3)) +
    stat_function(fun=dnorm, xlim=c(-3,3), args = list(mean=0, sd=0.7), color="blue") +
    ggtitle(paste0("hidden=", hidden_dim))
}


grobs <- lapply(c(1, 2, 5, 100), simulate_1d)
grid.arrange(grobs=grobs)



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


