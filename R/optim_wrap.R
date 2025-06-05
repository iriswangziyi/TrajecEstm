#' Low-level optimiser (direct C++ call)
#'
#' Exposes \code{estimate_beta_theta_lbfgs()} so it can be called
#' directly from R.  Most users should call [run_estimator()],
#' which adds default arguments and post-processing.
#'
#' @inheritParams run_estimator
#' @return Numeric vector: \eqn{(\hat\beta, \log\hat\theta)}.
#' @export
estimate_beta_theta_lbfgs <- TrajecEstm:::estimate_beta_theta_lbfgs
