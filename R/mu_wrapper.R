#' \code{mu_hat}: baseline mean estimator \hat{\mu}(t,s) for any scenario
#' @param j        integer {1,2}
#' @param t,s      evaluation time t and landmark s (same length if vectors)
#' @param h        bandwidth
#' @param par      numeric vector: c(beta_j, theta_j)
#' @param X,Y      design matrix (p x n) and response (n)
#' @param deltaPi  length-n integer vector with values in {1,2}
#' @param A,Z      length-n subject-level s and time vectors
#' @param scenario numeric, like 1.1, 1.2, 2.1, 2.2
#' @export
mu_hat <- function(j, t, s, h, par, X, Y, deltaPi, A, Z, scenario) {
    mu_r(j = j, t = t, a = s, h = h, btj = par,
         X = X, Y = Y, delPi = deltaPi, A = A, Z = Z, sce = scenario)
}

