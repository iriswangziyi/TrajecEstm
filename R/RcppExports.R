# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

PPL_sigmoid <- function(btj, j, X, Y_A, A, Z, Kmat, h1, tau0, tau1) {
    .Call(`_TrajecEstm_PPL_sigmoid`, btj, j, X, Y_A, A, Z, Kmat, h1, tau0, tau1)
}

estimate_beta_theta_lbfgs_V1 <- function(j, X, Y_A, A, Z, Kmat, tau0, tau1, init, tol = 1e-8, max_iter = 1000L) {
    .Call(`_TrajecEstm_estimate_beta_theta_lbfgs_V1`, j, X, Y_A, A, Z, Kmat, tau0, tau1, init, tol, max_iter)
}

estimate_beta_theta_lbfgs_V2 <- function(j, X, Y_A, A, Z, Kmat, tau0, tau1, init, tol = 1e-8, max_iter = 1000L) {
    .Call(`_TrajecEstm_estimate_beta_theta_lbfgs_V2`, j, X, Y_A, A, Z, Kmat, tau0, tau1, init, tol, max_iter)
}

PPL6_gamma <- function(j, btj, X, Y_A, A, Z, delPi, h1, tau0, tau1) {
    .Call(`_TrajecEstm_PPL6_gamma`, j, btj, X, Y_A, A, Z, delPi, h1, tau0, tau1)
}

PPL6_exp <- function(j, btj, X, Y_A, A, Z, delPi, h1, tau0, tau1) {
    .Call(`_TrajecEstm_PPL6_exp`, j, btj, X, Y_A, A, Z, delPi, h1, tau0, tau1)
}

PPL6_r <- function(j, btj, X, Y_A, A, Z, delPi, h1) {
    .Call(`_TrajecEstm_PPL6_r`, j, btj, X, Y_A, A, Z, delPi, h1)
}

mu6_gamma <- function(j, t, a, h, btj, X, Y, delPi, A, Z) {
    .Call(`_TrajecEstm_mu6_gamma`, j, t, a, h, btj, X, Y, delPi, A, Z)
}

mu6_exp <- function(j, t, a, h, btj, X, Y, delPi, A, Z) {
    .Call(`_TrajecEstm_mu6_exp`, j, t, a, h, btj, X, Y, delPi, A, Z)
}

mu6_sigmoid <- function(j, t, a, h, btj, X, Y, delPi, A, Z) {
    .Call(`_TrajecEstm_mu6_sigmoid`, j, t, a, h, btj, X, Y, delPi, A, Z)
}

mu6_r <- function(j, t, a, h, btj, X, Y, delPi, A, Z) {
    .Call(`_TrajecEstm_mu6_r`, j, t, a, h, btj, X, Y, delPi, A, Z)
}

matK <- function(Z, h1) {
    .Call(`_TrajecEstm_matK`, Z, h1)
}

matK_sparse <- function(Z, h1) {
    .Call(`_TrajecEstm_matK_sparse`, Z, h1)
}

matK_dispatch <- function(Z, h1, use_sparse) {
    .Call(`_TrajecEstm_matK_dispatch`, Z, h1, use_sparse)
}

compute_r_dr <- function(a, z, theta) {
    .Call(`_TrajecEstm_compute_r_dr`, a, z, theta)
}

gradi <- function(btj, j, X, Y_A, A, Z, Kmat, h1, tau0, tau1) {
    .Call(`_TrajecEstm_gradi`, btj, j, X, Y_A, A, Z, Kmat, h1, tau0, tau1)
}

rfun <- function(a, t, theta) {
    .Call(`_TrajecEstm_rfun`, a, t, theta)
}

rfun2 <- function(a, t, theta) {
    .Call(`_TrajecEstm_rfun2`, a, t, theta)
}

