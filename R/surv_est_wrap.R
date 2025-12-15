#' estimate_survivor_grad
#'   L-BFGS for survivor model (Section 4)
#'
#' @export
estimate_survivor_grad <- function(df_s, X_s,
                                   h1         = NULL,
                                   use_sparse = FALSE,
                                   tau0       = 0,
                                   tau        = Inf,
                                   tol        = 1e-8,
                                   max_iter   = 1000,
                                   init) {
    # df_s columns assumed:
    #   Y   : marker at baseline
    #   A   : baseline time
    #   Z   : event/censoring time
    # X_s  : p × n design matrix aligned with df_s rows/cols

    # 1. Filter for long-term survivors: Z >= tau
    keep <- df_s$Z >= tau
    df_s2 <- df_s[keep, , drop = FALSE]
    X_s2  <- X_s[, keep, drop = FALSE]

    n2 <- ncol(X_s2)
    if (is.null(h1)) h1 <- 2 * n2^(-1/3)

    # 2. Kernel on A *after* filtering (A is in the math)
    Kmat <- matK_dispatch(df_s2$A, h1, use_sparse)

    # 3. Call C++ optimizer (no Z passed anymore)
    par_cpp <- estimate_beta_survivor_lbfgs(
        X    = X_s2,
        Y    = df_s2$Y,
        A    = df_s2$A,
        Kmat = Kmat,
        tau0 = tau0,
        tau  = tau,
        init = init,
        tol  = tol,
        max_iter = max_iter
    )

    par_cpp
}
