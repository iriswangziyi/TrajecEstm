# Pure-R versions of the two estimation objectives, written from the paper's formulas and
# independent of the package's C++ code. The tests use them to check that the package's
# L-BFGS estimates are stationary points (and maximizers) of the paper's objective.
#
# Cause-specific model, eq. (obj1):
#   -log L(beta, theta) = - sum_{i,l} w_il [ beta'X_il + log r(S_il, Z_i; theta) - log D(Z_i) ],
#   D(t) = sum_{k,l'} I(0 <= S_kl') K_h(t - Z_k) exp{beta'X_kl' + log r(S_kl', Z_k; theta)},
#   w_il = I(0 <= S_il, tau0 <= Z_i <= tau1) Y_il,
# summed over the rows of subjects with an observed type-j failure (Delta_i = 1, Pi_i = j).
#
# Survivor model, eq. (surv_obj):
#   -log L(beta) = - sum_{i,l} w_il [ beta'X_il - log D(S_il) ],
#   D(s) = sum_{k,l'} K_h(s - S_kl') exp{beta'X_kl'},
#   w_il = I(tau0 <= S_il <= tau1) Y_il,
# summed over the rows of long-term survivors (Z_i >= tau).
#
# The factor 1/n inside s_hat only adds a constant to log L and is dropped.

# Second-order Epanechnikov kernel K_h(u) = 0.75 {1 - (u/h)^2} I(|u/h| <= 1) / h.
epan_kernel <- function(u, h) {
  x <- u / h
  0.75 * (1 - x^2) * (abs(x) <= 1) / h
}

# log r(s, t; theta) for the four shape codes, with s* = s / tau_norm and t* = t / tau_norm.
#   2.1  sigmoid:            r = {1 + exp[-theta1 (s* - t*/2)]}^(-1)
#   2.2  s + s^2:            log r = theta1 (s* - c)^2 + theta2 (s* - c)
#   1.1  s + s x t:          log r = theta1 (s* - c) + theta2 (s* - c)(t* - c)
#   1.2  (t-s) + (t-s)^2:    log r = theta1 (t* - s* - c)^2 + theta2 (t* - s* - c)
log_r_shape <- function(s, t, theta, code, tau_norm, center) {
  ss <- s / tau_norm
  ts <- t / tau_norm
  switch(as.character(code),
    "2.1" = -log1p(exp(-theta[1] * (ss - 0.5 * ts))),
    "2.2" = theta[1] * (ss - center)^2 + theta[2] * (ss - center),
    "1.1" = theta[1] * (ss - center) + theta[2] * (ss - center) * (ts - center),
    "1.2" = theta[1] * (ts - ss - center)^2 + theta[2] * (ts - ss - center),
    stop("unknown shape code ", code)
  )
}

n_theta <- function(code) if (code == 2.1) 1L else 2L

# d is a cause-specific setup from cause_setup(): X (p x m), Y, S, Z for the type-j rows,
# K[i, k] = K_h(Z_i - Z_k), tau0, tau1, tau_norm, code, center.
negloglik_cause <- function(par, d) {
  p <- nrow(d$X)
  beta <- par[seq_len(p)]
  theta <- par[-seq_len(p)]
  eta <- drop(crossprod(d$X, beta)) +
    log_r_shape(d$S, d$Z, theta, d$code, d$tau_norm, d$center)
  in_s <- as.numeric(d$S >= 0)
  w <- d$Y * in_s * (d$Z >= d$tau0 & d$Z <= d$tau1)
  den <- drop(d$K %*% (in_s * exp(eta)))
  pos <- w > 0
  -sum(w[pos] * (eta[pos] - log(den[pos])))
}

# d is a survivor setup from surv_setup(): X (p x m), Y, S for the rows with Z >= tau,
# K[i, k] = K_h(S_i - S_k), tau0, tau1.
negloglik_surv <- function(beta, d) {
  eta <- drop(crossprod(d$X, beta))
  w <- d$Y * (d$S >= d$tau0 & d$S <= d$tau1)
  den <- drop(d$K %*% exp(eta))
  pos <- w > 0
  -sum(w[pos] * (eta[pos] - log(den[pos])))
}

# Five-point central-difference gradient; step for coordinate k is step * (1 + |x_k|).
fd_grad <- function(f, x, step = 1e-3) {
  vapply(seq_along(x), function(k) {
    e <- rep(0, length(x))
    e[k] <- step * (1 + abs(x[k]))
    (8 * (f(x + e) - f(x - e)) - (f(x + 2 * e) - f(x - 2 * e))) / (12 * e[k])
  }, numeric(1))
}

# Central-difference Jacobian of a vector-valued function g at x (rows = outputs).
fd_jacobian <- function(g, x, step = 1e-6) {
  vapply(seq_along(x), function(k) {
    e <- rep(0, length(x))
    e[k] <- step * (1 + abs(x[k]))
    (g(x + e) - g(x - e)) / (2 * e[k])
  }, numeric(length(g(x))))
}

# Reference optimum of a pure-R objective: optim() BFGS from the same start as the package,
# using the finite-difference gradient of the objective itself.
optim_reference <- function(f, init) {
  stats::optim(init, f, gr = function(x) fd_grad(f, x, step = 1e-4),
               method = "BFGS", control = list(reltol = 1e-15, maxit = 1000))
}

# Size of the gradient at the estimate relative to the gradient at the starting value.
rel_grad <- function(g_hat, g_start) max(abs(g_hat)) / max(1, max(abs(g_start)))

# Tolerances. In a separate check of the C++ objectives (survivor objective with the corrected
# gradient) on simulated data with n = 1000 to 4000 and on two real-data markers, the
# finite-difference gradient at the package's estimates was below 1e-4 in absolute value, the
# estimates agreed with an independent Newton solution to within 1.2e-6, and the Jacobian of
# compute_r_dr() agreed with finite differences to within 2.2e-9. Here the gradient must shrink by a
# factor of 1e4 from the starting value, the estimate must agree with the optim() reference to
# 1e-4 in every coordinate, and the Jacobian must agree with finite differences to 1e-6.
GRAD_REL_TOL <- 1e-4
PAR_TOL <- 1e-4
JAC_REL_TOL <- 1e-6
