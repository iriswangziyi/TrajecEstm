// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu.h"
#include "utils.h"

// [[Rcpp::export]]
double mu6_gamma(arma::uword j,
                 double t,
                 double a,
                 double h,
                 const arma::vec& btj,
                 const arma::mat& X,
                 const arma::vec& Y,
                 const arma::uvec& delPi,
                 const arma::vec& A,
                 const arma::vec& Z) {

    int n = A.n_elem;
    int p = X.n_rows;

    Rcpp::Function dg("dgamma");
    Rcpp::Function pg("pgamma");

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    arma::vec theta = exp(btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem-1)));

    Rcpp::NumericVector gA = pg(A, theta(0));
    Rcpp::NumericVector gZ = pg(Z, theta(0));
    Rcpp::NumericVector r = gA;
    //Rcpp::NumericVector gA = dg(A, theta(0), theta(1));
    //Rcpp::NumericVector r = gA +theta(2);

    arma::vec xbj = X.t() * bj;

    //
    //   double gj = 0;
    //
    //
    //   for (int i=0; i<n; i++) {
    //     if (delPi(i)==j) {
    //       double den2 = 0;
    //       for (int k=0; k<n; k++) {
    //         if(delPi(k) == j)
    //         {
    //           den2 = den2 + ( 0.75 * std::max( 1-pow((Z(k)-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(k) ) * r[k];
    //         }
    //       }
    //       gj = gj + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i)) / den2;
    //     }
    //   }
    //
    double num = 0;
    double den = 0;
    for (int i=0; i<n; i++) {
        if (delPi(i)==j) {
            den = den + ( 0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(i) ) * r[i];
            num = num + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i));
        }
    }

    Rcpp::NumericVector da = pg(a, theta(0));
    Rcpp::NumericVector pt = pg(t, theta(0));
    Rcpp::NumericVector mu6 = da * num / den;

    // Rcpp::NumericVector da = dg(a, theta(0), theta(1));
    // Rcpp::NumericVector mu6 = ( da+theta(2) ) * num / den;
    return mu6[0];
}



/*
 mu_r: kernel-smoothed baseline mean estimator at (t, s) for trajectory j.

 where K(u) = (1 − u^2)·1{|u|<1} (Epanechnikov), beta_j and theta_j are packed in btj,
 and r(·,·; theta, sce) is scenario-specific.

 Args:
 j       : which trajectory (1 or 2)
 t, a    : evaluation time t and landmark s (named 'a' here)
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
            double a,
            double h,
            const arma::vec& btj,
            const arma::mat& X,
            const arma::vec& Y,
            const arma::uvec& delPi,
            const arma::vec& A,
            const arma::vec& Z,
            double sce)
{
    int n = A.n_elem;
    int p = X.n_rows;

    arma::vec bj = btj(arma::regspace<arma::uvec>(0, p - 1));
    arma::vec theta = btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem - 1));

    arma::vec xbj = X.t() * bj;

    // r_i = r(A_i, Z_i; theta) for each subject i under the given scenario
    arma::vec r = compute_r_vec(A, Z, theta, sce);

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

    // r(s, t; theta) at the evaluation point (s = a, t)
    double r_star = compute_r_scalar(a, t, theta, sce);

    // small numerical guard
    const double eps = 1e-12;
    if (den <= eps) return NA_REAL; // or 0.0 if prefer silent zero

    return (num / den) * r_star;
}


