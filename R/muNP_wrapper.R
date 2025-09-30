#' \code{muNP_hat}: baseline mean estimator \hat{\mu}(t,s)
#' @param j        integer {1,2}
#' @param t,s      evaluation time t and landmark s (same length if vectors)
#' @param h1        bandwidth1
#' @param h2        bandwidth2
#' @param par      numeric vector: c(beta_j)
#' @param X,Y      design matrix (p x n) and response (n)
#' @param deltaPi  length-n integer vector with values in {1,2}
#' @param A,Z      length-n subject-level s and time vectors
#' @export
muNP_hat <- function(j, t, s, h1,h2, par, X, Y, deltaPi, S, Z) {
    mu_NP(j = j, t = t, s = s, h1 = h1, h2 = h2, bj = par,
         X = X, Y = Y, delPi = deltaPi, S = S, Z = Z)
}

