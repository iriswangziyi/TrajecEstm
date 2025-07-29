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

        //TODO: note, delete comment later
    //when case of gamma, need theta > 0
    //i handle this in r, so no worry here

    arma::vec xbj = X.t() * bj;

    // generic r based on scenario
    arma::vec r = compute_r_vec(A, Z, theta, sce);

    double num = 0.0;
    double den = 0.0;

    if(t < h) {
        t = h;
    }

    for (int i = 0; i < n; i++) {
        double u = (t - Z(i)) / h;

        if (delPi(i)==j && std::abs(u) < 1.0) {
            double w = 1.0 - u * u;
            den += w * std::exp( xbj(i) ) * r(i);
            num += w * Y(i);
        }
    }

    // r at a specific (a, t)
    double r_star = compute_r_scalar(a, t, theta, sce);

    return (num / den) * r_star;
}


