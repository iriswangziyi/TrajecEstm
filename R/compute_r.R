#' Compute r(s, t; theta) under a scenario
#'
#' Low-level utility used for simulation and debugging. This wraps the
#' corresponding C++ implementation.
#'
#' The function computes
#' \deqn{r(s,t) = \exp\{\eta_\theta(s,t)\}}
#' where \eqn{\eta_\theta(s,t)} depends on `sce` (scenario code).
#'
#' @param s Numeric vector of s values.
#' @param t Numeric vector of t values (e.g., observed time Z).
#' @param theta Numeric vector of theta parameters.
#' @param sce Numeric scenario code (e.g., 1.1).
#' @param tau Numeric scaling constant used to normalize s and t inside
#'   \eqn{\eta_\theta(s,t)} (this is `tau_norm`, not the survival cutpoint).
#'
#' @return A numeric vector `r` of the same length as `s`.
#'
#' @examples
#' s <- seq(0, 10, length.out = 5)
#' t <- seq(2, 6, length.out = 5)
#' theta <- c(0.5, -1, 0.5)
#' compute_r_vec(s, t, theta, sce = 1.1, tau = 20)
#'
#' @export
compute_r_vec <- function(s, t, theta, sce, tau) {
    stopifnot(is.numeric(s), is.numeric(t), is.numeric(theta),
              is.numeric(sce), length(sce) == 1L,
              is.numeric(tau), length(tau) == 1L)

    if (length(s) != length(t)) {
        stop("`s` and `t` must have the same length.")
    }

    .Call(`_TrajecEstm_compute_r_vec`, s, t, theta, sce, tau)
}


#' Compute r(s, t; theta) and its derivatives with respect to theta
#'
#' Low-level utility used for gradient-based optimization and debugging.
#' Wraps the corresponding C++ implementation and returns both `r` and
#' the Jacobian matrix `dr`, where `dr[i,k] = d r_i / d theta_k`.
#'
#' @inheritParams compute_r_vec
#'
#' @return A list with:
#' \itemize{
#'   \item `r`: numeric vector of length `n`
#'   \item `dr`: numeric matrix of dimension `n x length(theta)`
#' }
#'
#' @examples
#' s <- seq(0, 10, length.out = 5)
#' t <- seq(2, 6, length.out = 5)
#' theta <- c(0.5, -1, 0.5)
#' out <- compute_r_dr(s, t, theta, sce = 1.1, tau = 20)
#' str(out)
#'
#' @export
compute_r_dr <- function(s, t, theta, sce, tau) {
    stopifnot(is.numeric(s), is.numeric(t), is.numeric(theta),
              is.numeric(sce), length(sce) == 1L,
              is.numeric(tau), length(tau) == 1L)

    if (length(s) != length(t)) {
        stop("`s` and `t` must have the same length.")
    }

    .Call(`_TrajecEstm_compute_r_dr`, s, t, theta, sce, tau)
}
