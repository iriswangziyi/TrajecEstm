
#' estimate_NP
#' @return estimation of beta
#
#'
#' @export

estimate_NP <- function(j, df_l, X_l, h1 = NULL, h2 = NULL,
                        d = 0, tau0 = 0, tau1 = Inf,
                        tol   = 1e-8, max_iter = 1000,
                        init) {
    keep <- df_l$delPi == j
    dfj_l <- df_l[keep, ]
    Xj_l <- X_l[, keep]
    n_j <- ncol(Xj_l)

    # Set default h if not supplied
    if (is.null(h1)) h1 <- 2 * n_j^(-1/7)
    if (is.null(h2)) h2 <- 2 * n_j^(-1/7)

    # Precompute kernel matrix
    Kmat <- matK_tri4_loop(dfj_l$Z, dfj_l$S, h1)

    #Kmat <- matK_tri4(dfj_l$Z, h1) * matK_tri4(dfj_l$S, h1)

    # Run optimization with gradient using RcppEnsmallen L_BFGS
    par_cpp <- estimate_beta_NP(
        j     = j,
        X     = Xj_l,
        Y_A   = dfj_l$Y,
        A     = dfj_l$S,
        Z     = dfj_l$Z,
        Kmat  = Kmat,
        d  = d,
        tau0  = tau0,
        tau1  = tau1,
        init  = init,         # in-place update
        tol   = 1e-8,         # ≈ MinGradientNorm()
        max_iter = 1000       # raise if needed
    )
}
