
#' estimate_ensmallen_grad
#' @return estimation stuff
#
#'
#' @export

estimate_ensmallen_grad <- function(j, df_l, X_l, h1 = NULL,
                                    use_sparse = FALSE,
                                    tau0 = 0, tau1 = Inf, tol   = 1e-8,
                                    max_iter = 1000) {
    keep <- df_l$delPi == j
    dfj_l <- df_l[keep, ]
    Xj_l <- X_l[, keep]
    n_j <- ncol(Xj_l)

    # Set default h1 if not supplied
    if (is.null(h1)) h1 <- 2 * n_j^(-1/3)

    # Precompute kernel matrix (dense or sparse)
    Kmat <- matK_dispatch(dfj_l$Z, h1, use_sparse)

    # Run optimization with gradient using RcppEnsmallen L_BFGS
    par_cpp <- estimate_beta_theta_lbfgs(
        j     = j,
        X     = Xj_l,
        Y_A   = dfj_l$Y,
        A     = dfj_l$S,
        Z     = dfj_l$Z,
        Kmat  = Kmat,
        tau0  = tau0,
        tau1  = tau1,
        init  = c(0, 0, 0),         # in-place update
        tol   = 1e-8,         # ≈ MinGradientNorm()
        max_iter = 1000       # raise if needed
    )
}
