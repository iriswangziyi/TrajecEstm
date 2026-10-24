
#' estimate_ensmallen_grad
#' @return estimation stuff
#
#'
#' @export

estimate_ensmallen_grad <- function(j, df_l, X_l, h1 = NULL,
                                    use_sparse = FALSE,
                                    tau0 = 0, tau1 = Inf, tau_norm,
                                    sce = 2.1,
                                    tol = 1e-8,
                                    max_iter = 1000,
                                    init,
                                    center = 0) {


    # Rows of type-j failures. Visits before the reference age (S < 0) are excluded: this is the
    # indicator I(0 <= A + V) in the paper's estimator (w_il, s-hat_jk, g-tilde_j).
    keep <- which(df_l$delPi == j & df_l$S >= 0)
    dfj_l <- df_l[keep, ]
    Xj_l <- X_l[, keep]
    n_j <- ncol(Xj_l)

    # Set default h1 if not supplied
    if (is.null(h1)) h1 <- 2 * n_j^(-1/3)

    # Precompute kernel matrix (dense or sparse)
    Kmat <- matK_dispatch(dfj_l$Z, h1, use_sparse)

    # Run optimization with gradient using RcppEnsmallen L_BFGS
    par_cpp <- estimate_beta_theta_lbfgs_V2(
        j     = j,
        X     = Xj_l,
        Y_S   = dfj_l$Y,
        S     = dfj_l$S,
        Z     = dfj_l$Z,
        Kmat  = Kmat,
        tau0  = tau0,
        tau1  = tau1,
        tau_norm = tau_norm,
        sce = sce,
        init  = init,
        tol   = 1e-8,
        max_iter = 1000,
        center = center
    )
}
