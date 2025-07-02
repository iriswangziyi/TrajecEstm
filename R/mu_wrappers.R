#' \code{mu6_sigmoid}: baseline mean estimator (sigmoid model)
#'
#' Direct wrapper around the C++ implementation. Most users will call
#' higher-level helpers, but this is exported for completeness.
#' @export
mu_sigmoid <- function(j, t, a, h, btj, X, Y, delPi, A, Z, sce){
    mu_r(j, t, a, h, btj, X, Y, delPi, A, Z, sce)
    }
