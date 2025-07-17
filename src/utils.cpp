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
double compute_r_scalar(double s,
                        double t,
                        arma::vec theta,
                        double sce)
{
    double r;

    if (sce == 2.1) {
        // Sigmoid
        r = 1.0 / (1.0 + std::exp(-theta(0) * (s - 0.5 * t)));

    } else if (sce == 2.2) {
        // Weibull CDF
        double base = s / (exp(theta(0)) * t);
        r = 1.0 - std::exp(-std::pow(base, exp(theta(1))));

    } else if (sce == 1.1) {
        // Gamma PDF: shape=3, scale=θ·t
        // theta need to be pos
        double shape = 3.0;
        double scale = exp(theta(0)) * t;
        r = std::pow(s, shape - 1.0) * std::exp(-s / scale) /
            (std::tgamma(shape) * std::pow(scale, shape));

    } else if (sce == 1.2) {
        // Polynomial
        double diff = s - 0.5 * t;
        r = theta(0) * diff * diff + theta(1);

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_scalar()");
    }

    return r;
}



//compute_r + compute_dr
// [[Rcpp::export]]
arma::vec compute_r_vec(arma::vec s,
                        arma::vec t,
                        arma::vec theta,
                        double sce)
{
    arma::vec r(s.n_elem);

    if (sce == 2.1) {
        // === Sigmoid: r(s, t) = 1 / (1 + exp(-θ(s - 0.5t))) ===
        r = 1 / (1 + exp(-theta(0) * (s - 0.5 * t)));

    } else if (sce == 2.2) {
        // === Weibull CDF: r(s,t) = 1 - exp( - (s / (θ₁·t))^θ₂ ) ===
        // shape and scale need to be pos
        arma::vec base = s / (exp(theta(0)) * t);
        r = 1 - exp(-pow(base, exp(theta(1))));

    } else if (sce == 1.1) {
        // === Gamma PDF: r(s,t) = dgamma(s; shape=3, scale=θ·t) ===
        // theta need to be pos
        double shape = 3.0;
        arma::vec scale = exp(theta(0)) * t;
        r = pow(s, shape - 1.0) % exp(-s / scale) / (tgamma(shape) * pow(scale, shape));

    } else if (sce == 1.2) {
        // === Polynomial: r(s,t) = θ1 * (s - 0.5t)^2 + θ2 ===
        arma::vec diff = s - 0.5 * t;
        r = theta(0) * diff % diff+ theta(1);

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_vec()");
    }

    return r;
}

//only have sigmoiod case, sce=2, for now
//r  = 1 / (1 + exp(theta(0) * (a - 0.5 * z)));
//dr = r % (1 - r) % (a - 0.5 * z); // d r / d theta


// [[Rcpp::export]]
Rcpp::List compute_r_dr(arma::vec s,
                        arma::vec t,
                        arma::vec theta,
                        double sce)
{
    int n = s.n_elem;
    arma::vec r(n);
    arma::mat dr(n, theta.n_elem, arma::fill::zeros); // n x #theta

    if (sce == 2.1) {
        // === Sigmoid: r(s, t; theta) = 1 / (1 + exp(-theta (s - 0.5t))) ===
        r = 1 / (1 + exp(-theta(0) * (s - 0.5 * t)));
        dr.col(0) = r % (1 - r) % (s - 0.5 * t);

    } else if (sce == 2.2) {
        // === Weibull CDF: r(s,t) = 1 - exp( - (s / (theta1·t))^theta2 ) ===
        // thetas need to be pos
        arma::vec base = s / (exp(theta(0)) * t);
        arma::vec power = pow(base, exp(theta(1)));
        arma::vec exp_term = exp(-power);
        r = 1 - exp_term;

        // dr/dtheta1 = exp_term * theta2 * (s/(theta1·t))^theta2 / theta1
        dr.col(0) = exp_term % power * exp(theta(1)) / exp(theta(0)) * exp(theta(0));

        // dr/dtheta2 = -log(s/(theta1·t)) * (s/(theta1·t))^theta2 * exp_term
        dr.col(1) = -log(base) % power % exp_term * exp(theta(1));

    } else if (sce == 1.1) {
        // === Gamma PDF: r(s,t) = dgamma(s; shape=3, scale=theta·t) ===
        // theta need to be pos
        double shape = 3.0;
        arma::vec scale = exp(theta(0)) * t;
        arma::vec coef = pow(s, shape - 1) % exp(-s / scale) / (tgamma(shape) * pow(scale, shape));
        r = coef;

        // dr/dtheta = d/dtheta [dgamma(s, shape=3, scale=theta·t)]
        // dr = r * [s / (scale^2) - shape / scale] * dscale/dtheta = r * (s - 3·scale) / (scale^2) * t
        dr.col(0) = r % (s - shape * scale) / (scale % scale) % t * exp(theta(0));

    } else if (sce == 1.2) {
        // === Poly: r(s,t) = theta1 * (s - 0.5t)^2 + theta2 ===
        arma::vec quad = (s - 0.5 * t)%(s - 0.5 * t);
        r = theta(0) * quad + theta(1);
        dr.col(0) = quad;
        dr.col(1) = arma::ones<arma::vec>(n);

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_dr()");
    }

    return Rcpp::List::create(
        Rcpp::Named("r")  = r,
        Rcpp::Named("dr") = dr
    );
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

    arma::vec xbj = X.t() * beta;       // Xbeta
    arma::vec exp_xbj = arma::exp(xbj);
    arma::vec r = 1 / (1 + arma::exp(theta * (A - Z / 2)));
    //TODO
    // We used arma::vec before and worked, need to check on this some time
    //actually should be arma::mat in general cases
    arma::vec dr = - r % (1 - r) % (A - Z / 2); // d r / d theta

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
