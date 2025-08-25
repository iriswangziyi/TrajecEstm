// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu_NP.h"
#include "utils.h"

/*
    mu_NP: kernel-smoothed baseline mean estimator at (t, s) for trajectory j.

where K_h(),  Fourth-Order Triweight
 (1/h) * (315/512) * (3 - 11 u^2) * (1 - u^2)^3 * I(|u|<=1)

Args:
    j       : which trajectory (1 or 2)
t, a    : evaluation time t and landmark s (named 'a' here)
h1       : bandwidth for Kh(Zi-Zk)
h2       : bandwidth for Kh(Ai+Vil-(Ak+Vkq))
bj     : concatenated parameters c(beta_j)
X (p×n) : design matrix (rows covariates, cols subjects)
Y (n)   : response
delPi   : del_i in {1,2}, length n
A, Z    : subject-level s_i and time Z_i (length n)

Returns:
    scalar \hat{mu}_j(t, s).
*/

// [[Rcpp::export]]
double mu_NP(arma::uword j,
            double t,
            double a,
            double h1,
            double h2,
            const arma::vec& bj,
            const arma::mat& X,
            const arma::vec& Y,
            const arma::uvec& delPi,
            const arma::vec& A,
            const arma::vec& Z)
{
    int n = A.n_elem;

    arma::vec xbj = X.t() * bj;

    double num = 0.0;
    double den = 0.0;

    for (int i = 0; i < n; i++) {
        double u1sqr = (t - Z(i)) / h1 * (t - Z(i)) / h1;
        double u2sqr = (a - A(i)) / h2 * (a - A(i)) / h2;

        if (delPi(i)==j && u1sqr < 1.0 && u1sqr < 1.0) {
            // 4th Triweight kernel support
            double w = (3.0 - 11.0 * u1sqr) * (1.0 -  u1sqr) *
                (1.0 -  u1sqr) * (1.0 -  u1sqr) *
                (3.0 - 11.0 * u2sqr) * (1.0 -  u2sqr) *
                (1.0 -  u2sqr) * (1.0 -  u2sqr);

            den += w * std::exp(xbj(i));
            num += w * Y(i);
        }
    }

    return (num / den);
}


