// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu_surv.h"
#include "utils.h"

/*
 mu_surv: kernel-smoothed survivor mean estimator at landmark s.

 Implements the "blue" discrete-sum version:
 μ̂(s;β) =
 [ Σ_rows K_h(s - S_row) I(Z_row ≥ τ) Y_row ]
 / [ Σ_rows K_h(s - S_row) I(Z_row ≥ τ) exp(βᵀ X_row) ].

 Notes:
 - Long-format: each row corresponds to one visit time S = A_i + V_il.
 - If you pre-filter to Z ≥ τ in the wrapper, the indicator is redundant.
 - Kernel convention matches your mu_r: weight = (1 - u^2) with support |u|<1.

 Args:
 s        : evaluation landmark
 h        : bandwidth
 beta     : p-vector
 X (p×n)  : design matrix (cols align with rows of Y/S/Z)
 Y (n)    : response at each row/visit
 S (n)    : visit time (A_i + V_il)
 Z (n)    : event/censoring time replicated per row
 tau      : survivor threshold

 Returns:
 scalar μ̂(s;β).
 */
// [[Rcpp::export]]
double mu_surv(double s,
               double h,
               const arma::vec& beta,
               const arma::mat& X,
               const arma::vec& Y,
               const arma::vec& S,
               const arma::vec& Z,
               double tau)
{
    const int n = S.n_elem;

    if (X.n_cols != (arma::uword)n)
        Rcpp::stop("X.n_cols must equal length(S).");
    if ((int)Y.n_elem != n || (int)Z.n_elem != n)
        Rcpp::stop("Y and Z must have same length as S.");

    arma::vec xb = X.t() * beta;   // n×1

    double num = 0.0;
    double den = 0.0;

    for (int i = 0; i < n; ++i) {
        // Survivor indicator (you can comment out if wrapper prefilters Z>=tau)
        if (Z(i) < tau) continue;

        double u = (s - S(i)) / h;
        if (std::abs(u) < 1.0) {
            double w = 1.0 - u * u;   // Epanechnikov support (same style as mu_r)
            num += w * Y(i);
            den += w * std::exp(xb(i));
        }
    }

    const double eps = 1e-12;
    if (den <= eps) return NA_REAL;

    return num / den;
}

// [[Rcpp::export]]
double mu_surv_core(double s,
                    double h,
                    const arma::vec& beta,
                    const arma::mat& X,
                    const arma::vec& Y,
                    const arma::vec& S)
{
    const arma::uword n = S.n_elem;

    if (X.n_cols != n) Rcpp::stop("X.n_cols must equal length(S).");
    if (Y.n_elem != n) Rcpp::stop("Y must have same length as S.");

    arma::vec xb = X.t() * beta;

    double num = 0.0, den = 0.0;

    for (arma::uword i = 0; i < n; ++i) {
        double u = (s - S(i)) / h;
        if (std::abs(u) < 1.0) {
            double w = 1.0 - u * u;
            num += w * Y(i);
            den += w * std::exp(xb(i));
        }
    }

    const double eps = 1e-12;
    if (den <= eps) return NA_REAL;
    return num / den;
}

