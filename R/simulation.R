#' Run one dataset → (beta, log-theta) estimates
#'
#' A thin R wrapper around the C++ L-BFGS optimiser.
#'
#' @inheritParams estimate_beta_theta_lbfgs
#' @return numeric vector: c(beta, logTheta)
#' @export
run_estimator <- function(j, X, Y_A, A, Z, Kmat,
                          tau0 = 0, tau1 = Inf,
                          init = rep(0, nrow(X) + 1),
                          tol  = 1e-8, max_iter = 1000) {
    estimate_beta_theta_lbfgs(j, X, Y_A, A, Z, Kmat,
                              tau0, tau1, init, tol, max_iter)
}
