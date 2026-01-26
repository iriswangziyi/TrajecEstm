#' estimate_survivor_grad
#'   L-BFGS for survivor model
#'
#' @export
estimate_survivor_grad <- function(df_s, X_s,
                                   h1         = NULL,
                                   use_sparse = FALSE,
                                   tau        = NULL,
                                   tau0       = NULL,
                                   tau1       = NULL,
                                   tol        = 1e-8,
                                   max_iter   = 1000,
                                   init) {
    # df_s columns assumed:
    #   Y : marker val
    #   S : measurement time (baseline case is fine)
    #   Z : event/censoring time
    # X_s: p × n matrix aligned with df_s columns (= subjects)

    # ---- Required: tau (for survivor filtering) ----
    if (is.null(tau)) {
        stop("`tau` is required for survivor model: it defines long survivors via Z >= tau.")
    }

    # ---- Optional: tau0/tau1 for boundary handling on S ----
    # Keep permissive fallback if missing, but warn loudly.
    if (is.null(tau0) || is.null(tau1)) {
        warning("`tau0`/`tau1` not provided. Using tau0=0 and tau1=Inf (no boundary restriction).")
        if (is.null(tau0)) tau0 <- 0
        if (is.null(tau1)) tau1 <- Inf
    }

    # sanity checks
    if (!is.finite(tau0) && tau0 != -Inf) stop("`tau0` must be finite (or -Inf).")
    if (tau1 != Inf && !is.finite(tau1)) stop("`tau1` must be finite (or Inf).")
    if (tau0 >= tau1) stop("Need tau0 < tau1.")

    # 1) Filter for long-term survivors: Z >= tau
    keep <- df_s$Z >= tau
    df_s2 <- df_s[keep, , drop = FALSE]
    X_s2  <- X_s[, keep, drop = FALSE]

    n2 <- ncol(X_s2)
    if (n2 == 0L) stop("No survivors after filtering Z >= tau. Check tau or sim settings.")

    # 2) Bandwidth h1 (kernel on S)
    if (is.null(h1)) {
        warning("`h1` not provided. Using fallback h1 = 2 * n^(-1/3) based on survivor sample size.")
        h1 <- 2 * n2^(-1/3)
    }

    # 3) Kernel matrix on S (after filtering)
    Kmat <- matK_dispatch(df_s2$S, h1, use_sparse)

    # 4) Call C++ optimizer
    par_cpp <- estimate_beta_survivor_lbfgs(
        X    = X_s2,
        Y    = df_s2$Y,
        S    = df_s2$S,     # <-- rename A -> S
        Kmat = Kmat,
        tau0 = tau0,
        tau1 = tau1,
        init = init,
        tol  = tol,
        max_iter = max_iter
    )

    par_cpp
}
