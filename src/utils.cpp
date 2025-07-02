// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include "utils.h"

// --------------------
// Kernel Matrix Helper
// --------------------
// [[Rcpp::export]]
arma::mat matK(const arma::vec& Z,
               double h1) {

    int n = Z.n_elem;

    // Step 1: Compute Z differences normally
    arma::mat Z_diff = arma::repmat(Z, 1, n) - arma::repmat(Z.t(), n, 1);
    //Rcpp::Rcout << Z_diff;

    // Step 2: Compute scaled differences
    arma::mat U = Z_diff / h1;

    // Step 3: Mask U before computing K (skip invalid kernel regions)
    // U.elem(arma::abs(U) >= 1).zeros();

    // Step 4: Compute kernel values **only for valid U**
    arma::mat K = 0.75 * (1 - arma::square(U)) % (arma::abs(U) <= 1)/ h1;

    return K;
}

// [[Rcpp::export]]
arma::sp_mat matK_sparse(const arma::vec& Z, double h1) {
    int n = Z.n_elem;
    arma::sp_mat K(n, n);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double u = (Z(i) - Z(k)) / h1;
            if (std::abs(u) < 1.0) {
                K(i, k) = 0.75 * (1 - u * u) / h1;
            }
        }
    }
    return K;
}

// [[Rcpp::export]]
arma::mat matK_dispatch(const arma::vec& Z, double h1,
                        Rcpp::LogicalVector use_sparse) {
    if (use_sparse) {
        arma::sp_mat Ksparse = matK_sparse(Z, h1);
        return arma::mat(Ksparse);  // Convert to dense for downstream use
    } else {
        return matK(Z, h1);
    }
}

//NEED return double's r
// [[Rcpp::export]]
double compute_r_scalar(double a,
                        double z,
                        arma::vec theta,
                        double sce)
{
    double r;

    if(sce == 2) {
        r = 1.0 / (1.0 + std::exp(theta(0) * (a - 0.5 * z)));
    } else {
        // placeholder: reuse sigmoid
        r = 1.0 / (1.0 + std::exp(theta(0) * (a - 0.5 * z)));
    }

    return r;
}


// [[Rcpp::export]]
// might not need, then take out...
Rcpp::List compute_r_dr_scalar(double a,
                               double z,
                               arma::vec theta,
                               double sce)
{
    double r, dr;

    if(sce == 2) {
        r  = 1.0 / (1.0 + std::exp(theta(0) * (a - 0.5 * z)));
        dr = -r * (1.0 - r) * (a - 0.5 * z);
    } else {
        // placeholder: reuse sigmoid
        r  = 1.0 / (1.0 + std::exp(theta(0) * (a - 0.5 * z)));
        dr = -r * (1.0 - r) * (a - 0.5 * z);
    }

    return Rcpp::List::create(
        Rcpp::Named("r")  = r,
        Rcpp::Named("dr") = dr);
}


//compute_r + compute_dr
// [[Rcpp::export]]
arma::vec compute_r_vec(arma::vec a,
                        arma::vec z,
                        arma::vec theta,
                        double sce)
{
    arma::vec r;

    if(sce == 2) {
        r = 1 / (1 + exp(theta(0) * (a - 0.5 * z)));
    } else {
        // placeholder: reuse sigmoid
        r = 1 / (1 + exp(theta(0) * (a - 0.5 * z)));
    }

    return r;
}



// [[Rcpp::export]]
Rcpp::List compute_r_dr(arma::vec a,
                        arma::vec z,
                        arma::vec theta,
                        double sce)
{
    arma::vec r;
    arma::vec dr;

    //only have sigmoiod case, sce=2, for now
    if(sce == 2) {
        r  = 1 / (1 + exp(theta(0) * (a - 0.5 * z)));
        dr = -r % (1 - r) % (a - 0.5 * z); // d r / d theta
    } else {
        // placeholder: reuse sigmoid
        //TODO: gamma's r and dr
        r  = 1 / (1 + exp(theta(0) * (a - 0.5 * z)));
        dr = -r % (1 - r) % (a - 0.5 * z);
    }

    return Rcpp::List::create(
        Rcpp::Named("r")  = r,
        Rcpp::Named("dr") = dr);
}


// --------------------
// Gradient function
// --------------------
// [[Rcpp::export]]
arma::vec gradi(arma::vec btj,
                arma::uword j,
                const arma::mat& X,
                const arma::vec& Y_A,
                const arma::vec& A,
                const arma::vec& Z,
                const arma::mat& Kmat,
                double h1,
                double tau0,
                double tau1) {

    int n = A.n_elem;
    int p = X.n_rows;  // p = 2 for beta
    arma::vec grad_beta(p, arma::fill::zeros);
    double grad_theta = 0.0;

    arma::vec beta = btj.subvec(0, p - 1);
    double theta = std::exp(btj(p));  // note: we optimize log(theta)

    arma::vec xbj = X.t() * beta;       // Xβ
    arma::vec exp_xbj = arma::exp(xbj);
    arma::vec r = 1 / (1 + arma::exp(theta * (A - Z / 2)));
    //TODO
    // We used arma::vec before and worked, need to check on this some time
    //actually should be arma::mat in general cases
    arma::vec dr = -r % (1 - r) % (A - Z / 2); // d r / d theta

    for (int i = 0; i < n; ++i) {
        if (Z(i) >= tau0 && Z(i) <= tau1) {
            double yi = Y_A(i);
            //arma::rowvec ki = Kmat.row(i); //apply column-access optimization
            arma::vec ki = Kmat.col(i);  // Kmat is symmetric

            //arma::vec weight = ki.t() % exp_xbj % r;
            arma::vec weight = ki % exp_xbj % r;

            double denom = arma::sum(weight);

            // Gradient for beta
            //arma::vec num_beta = ki.t() % exp_xbj % r;
            arma::vec weighted_X = X * weight;
            arma::vec grad_i_beta = X.col(i) - weighted_X / denom;

            // Gradient for theta
            double dlogr_i = dr(i) / r(i);
            //double weighted_dlogr = arma::sum(ki.t() % exp_xbj % dr) / denom;

            //TODO
            //Might have a problem for general dr,
            double weighted_dlogr = arma::sum(ki % exp_xbj % dr) / denom;
            double grad_i_theta = dlogr_i - weighted_dlogr;

            // Update total gradient
            grad_beta += yi * grad_i_beta;
            grad_theta += yi * grad_i_theta;
        }
    }

    arma::vec grad(p + 1);
    grad.subvec(0, p - 1) = grad_beta;
    grad(p) = grad_theta * theta;  // chain rule for log(theta)

    return -grad;  // minus for optimization (minimize -logPPL)
}

// --------------------
// rfun utilities
// --------------------
// [[Rcpp::export]]
arma::vec rfun(arma::vec a,
               arma::vec t,
               arma::vec theta)
{
    //return 1/(1+ exp(  (theta(0)*t+theta(1)) % (a - (theta(2)*t+theta(3)) )));
    return exp(theta(3)) + 1/(1+ exp(  (theta(0)) * (a - (theta(1)*t+theta(2)) )));

    //return 1/(1+ exp(  (theta(0)*t+theta(1)) % (a - (theta(2)*t+theta(3)) )));
}

// [[Rcpp::export]]
double rfun2(double a,
             double t,
             arma::vec theta)
{
    //return 1/(1+ exp(  (theta(0)*t+theta(1))  *(a -  (theta(2)*t+theta(3))   )));
    return exp(theta(3)) + 1/(1+ exp(  (theta(0)) * (a - (theta(1)*t+theta(2)) )));


}
