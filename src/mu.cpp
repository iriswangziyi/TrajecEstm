// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu.h"
#include "utils.h"

/*
 mu_r: kernel-smoothed baseline mean estimator at (t, s) for trajectory j.

 where K(u) = (1 − u^2)·1{|u|<1} (Epanechnikov), beta_j and theta_j are packed in btj,
 and r(·,·; theta, sce) is scenario-specific.

 Args:
 j       : which trajectory (1 or 2)
 t, s    : evaluation time t and landmark s
 h       : bandwidth
 btj     : concatenated parameters c(beta_j, theta_j)
 X (p×n) : design matrix (rows covariates, cols subjects)
 Y (n)   : response
 delPi   : del_i in {1,2}, length n
 A, Z    : subject-level s_i and time Z_i (length n)
 sce     : scenario code (1.1, 1.2, 2.1, 2.2)

 Returns:
 scalar \hat{mu}_j(t, s).
 */
// [[Rcpp::export]]
double mu_r(arma::uword j,
            double t,
            double s,
            double h,
            const arma::vec& btj,
            const arma::mat& X,
            const arma::vec& Y,
            const arma::uvec& delPi,
            const arma::vec& S,
            const arma::vec& Z,
            double sce)
{
    int n = S.n_elem;
    int p = X.n_rows;

    arma::vec bj = btj(arma::regspace<arma::uvec>(0, p - 1));
    arma::vec theta = btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem - 1));

    arma::vec xbj = X.t() * bj;

    // r_i = r(S_i, Z_i; theta) for each subject i under the given scenario
    arma::vec r = compute_r_vec(S, Z, theta, sce);

    double num = 0.0;
    double den = 0.0;

    if(t < h) t = h;

    for (int i = 0; i < n; i++) {
        double u = (t - Z(i)) / h;

        if (delPi(i)==j && std::abs(u) < 1.0) {
            // Epanechnikov kernel support
            double w = 1.0 - u * u;
            den += w * std::exp( xbj(i) ) * r(i);
            num += w * Y(i);
        }
    }

    // r(s, t; theta) at the evaluation point (s, t)
    double r_star = compute_r_scalar(s, t, theta, sce);

    // small numerical guard
    const double eps = 1e-12;
    if (den <= eps) return NA_REAL; // or 0.0 if prefer silent zero

    return (num / den) * r_star;
}


