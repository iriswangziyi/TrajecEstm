# r(s,t;theta)

library(ggplot2)
library(dplyr)
library(tidyr)

# === Setup s, t grid ===
#generate t, s
t_grid = 1:30
s_grid = lapply(t_grid, function(t) seq(0, t, by = 0.1))

r_grid <- data.frame(
    t = rep(t_grid, 10*(1:30) + 1),
    s = unlist(s_grid))
tau = 18


rfun_polt <- function(s, t, theta0, theta1, theta2) {
    exp(theta0 * (s/tau) * (s/tau)
        + theta1 * (s/tau)
        + theta2 * (s/tau) * (t/tau))
}

r_grid$rfun_polt <- rfun_polt(r_grid$s, r_grid$t, 
                              theta0=0.5, theta1 = 1, theta2 = 1)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
p_poly2 <- ggplot(r_grid, aes(x = s, y = rfun_polt, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

r_grid$rfun_output <- rfun_polt(r_grid$s, r_grid$t, 
                              theta0=1, theta1 = 1.5, theta2 = -0.5)
p_poly3 <- ggplot(r_grid, aes(x = s, y = rfun_output, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

r_grid$what <- rfun_polt(r_grid$s, r_grid$t, 
                              theta0=0.5, theta1 = -0.5, theta2 = 0.5)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
ggplot(r_grid, aes(x = s, y = what, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")


r_grid$try1 <- rfun_polt(r_grid$s, r_grid$t, 
                         theta0=0.1, theta1 = -0.5, theta2 = 0.5)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
ggplot(r_grid, aes(x = s, y = try1, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

r_grid$try2 <- rfun_polt(r_grid$s, r_grid$t, 
                         theta0=0.1, theta1 = -0.5, theta2 = 1)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
ggplot(r_grid, aes(x = s, y = try2, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

r_grid$try7 <- rfun_polt(r_grid$s, r_grid$t, 
                         theta0=0.5, theta1 = -1, theta2 = 0.5)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
ggplot(r_grid, aes(x = s, y = try7, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

r_grid$simput2 <- rfun_polt(r_grid$s, r_grid$t, 
                         theta0=0.79470976, theta1 = -0.89643115, theta2 = -0.06117081)
#g_poly2 <- function(t) 1.5 + 0.05 * t
#r_grid$mu_poly2 <- r_grid$r_poly2 * g_poly2(r_grid$t)
ggplot(r_grid, aes(x = s, y = simput2, group = t)) +
    geom_line(aes(color = t)) +
    labs(title = "r", y = "r(s,t)")

#2.10
#100 sample of n=10000, no BT
#check colMean, if mean close to truth -> ?
#if mean 