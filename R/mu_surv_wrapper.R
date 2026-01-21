#' \code{mu_surv_hat}: survivor baseline mean estimator \hat{\mu}_0(s; \beta)
#'
#' Kernel-smoothed estimator for the survivor/healthy group mean function \eqn{\mu_0(s)}.
#' This implements the discrete-sum ("blue") version on long-format data:
#' \deqn{
#' \hat{\mu}_0(s;\beta)=
#' \frac{\sum_{r=1}^n K_h(s-S_r)\, I(Z_r\ge\tau)\, Y_r}
#' {\sum_{r=1}^n K_h(s-S_r)\, I(Z_r\ge\tau)\, \exp\{\beta^\top X_r\}}
#' }
#'
#' Each row \eqn{r} corresponds to one observed visit time \eqn{S_r = A_i + V_{il}}
#' in long-format data, with marker value \eqn{Y_r} and covariates \eqn{X_r}.
#'
#' @param s     evaluation landmark(s) (numeric scalar or vector)
#' @param h     bandwidth
#' @param beta  numeric vector of regression coefficients (length p)
#' @param X     design matrix (p x n), columns aligned with Y/S/Z
#' @param Y     length-n marker values (one per row/visit)
#' @param S     length-n visit times (one per row/visit)
#' @param Z     length-n event/censoring times replicated per row
#' @param tau   survivor threshold \eqn{\tau}
#'
#' @return numeric vector of \eqn{\hat{\mu}_0(s;\beta)} with the same length as \code{s}
#' @export
mu_surv_hat <- function(s, h, beta, X, Y, S, Z, tau) {
    vapply(
        s,
        function(si) mu_surv(si, h = h, beta = beta, X = X, Y = Y, S = S, Z = Z, tau = tau),
        numeric(1)
    )
}

#' @export
mu_surv_hat_fast <- function(s, h, beta, X, Y, S, Z, tau) {
    keep <- (Z >= tau)
    if (!any(keep)) return(rep(NA_real_, length(s)))

    Xk <- X[, keep, drop = FALSE]
    Yk <- Y[keep]
    Sk <- S[keep]

    vapply(
        s,
        function(si) mu_surv_core(si, h = h, beta = beta, X = Xk, Y = Yk, S = Sk),
        numeric(1)
    )
}
