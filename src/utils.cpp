// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include "utils.h"

// --------------------
// Kernel Matrix Helper for Model II
// K_h(Z_k-Z_i),  Fourth-Order Triweight
// u = (Z_k-Z_i)/h, K_h = 1/h * k(u),
// (1/h) * (315/512) * (3 - 11 u^2) * (1 - u^2)^3 * I(|u|<=1)
// --------------------
// [[Rcpp::export]]
arma::mat matK_tri4(const arma::vec& Z,
                    double h) {

    int n = Z.n_elem;

    // Pairwise differences: Z_i - Z_k
    arma::mat Zmat = arma::repmat(Z, 1, n);
    arma::mat U2   = arma::square((Zmat - Zmat.t())/h);

    //we can make this faster, by implementing a loop
    //for every U2's elem > 1, mat K's elem = 0

    // Compute kernel values **only for valid U**
    arma::mat K = (315.0 / 512.0) * (3.0 - 11.0 * U2) %
        arma::pow((1.0 - U2), 3) % (U2 <= 1)/ h;

    return K;
}

//Check performance
//if using loop structure, should write both kernel together
// [[Rcpp::export]]
arma::mat matK_tri4_loop(const arma::vec& Z,
                          const arma::vec& S,
                          double h) {

    const int n = Z.n_elem;
    arma::mat K(n, n, arma::fill::zeros);

    const double inv_h = 1.0 / h;
    const double c = 315.0 / 512.0;

    double diagVal2 = c * 3.0 * inv_h * c * 3.0 * inv_h;


    for (int i = 0; i < n; ++i) {
        // diagonal: u = 0 => (3 - 11*0)*(1-0)^3 = 3
        K(i, i) = diagVal2;

        for (int k = i + 1; k < n; ++k) {
            double d = (Z(i) - Z(k)) * inv_h;  // u = (Zi - Zk)/h
            double ds = (S(i) - S(k)) * inv_h;  // ds = (Si - Sk)/h

            // outside support > 0
            if (std::abs(d) > 1.0 || std::abs(ds) > 1.0) continue;

            double u2 = d * d;
            double ds2 = ds * ds;

            double omu2 = 1.0 - u2;           // (1 - u^2)
            double omds2 = 1.0 - ds2;

            double val  = c * (3.0 - 11.0 * u2) * omu2 * omu2 * omu2 * inv_h *
                c * (3.0 - 11.0 * ds2) * omds2 * omds2 * omds2 * inv_h;

            K(i, k) = val;
            K(k, i) = val;                          // symmetry
        }
    }
    return K;
}


// if we are doing K_h(A_k+ V_{kq}-A_i+V_{il}),  Fourth-Order Triweight
// our Z_k just be A_k+ V_{kq}, so we can still use this matK_tri4(A_k+ V_{kq}, h)


// --------------------
// Kernel Matrix Helper for Model I
// K_h(Z_k-Z_i),  Second-Order Epanechnikov
// --------------------
// [[Rcpp::export]]
arma::mat matK(const arma::vec& Z,
               double h1) {

    int n = Z.n_elem;

    // Compute Z differences normally
    arma::mat U = (arma::repmat(Z, 1, n) - arma::repmat(Z.t(), n, 1))/h1;

    // idea: Mask U before computing K (skip invalid kernel regions)
    // U.elem(arma::abs(U) >= 1).zeros();

    // Compute kernel values **only for valid U**
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


/*
 compute_r_scalar: scalar r(s, t; θ, sce) at a single (s, t).

 Let s* = s/tau and t* = t/tau with tau = 20.0 (fixed scaling).

 Scenarios:
 2.1 (Sigmoid):
 r = 1 / (1 + exp( -theta1 · (s* − 0.5 · t*) ))

 2.2 (Polynomial in s only):
 log r = theta1 · s* + theta2 · (s*)^2     // no t dependence

 1.1 (Bell / bump with shift):
 log r = −theta1 · (s* − 0.5 · t*)^2 + θ2 · s*

 1.2 (General polynomial with interaction):
 log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*
 */
// [[Rcpp::export]]
double compute_r_scalar(double s,
                        double t,
                        arma::vec theta,
                        double sce)
{
    double r;
    double tau = 20.0;

    if (sce == 2.1) {
        // Sigmoid shape centered near s* = 0.5 · t*
        r = 1.0 / (1.0 + std::exp(-theta(0) * ((s/tau) - 0.5 * (t/tau))));

    } else if (sce == 2.2) {
        // Polynomial in s only (no t):
        //Poly2:log r = theta1 · s* + theta2 · (s*)^2
        r = std::exp(theta(0) * (s/tau) + theta(1) * (s/tau) *(s/tau));

    } else if (sce == 1.1) {
        // Bell:
        //Poly3:log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        double diff = (s/tau) - 0.5 * (t/tau);
        r = std::exp(-theta(0) * diff * diff + theta(1) * (s/tau));

    } else if (sce == 1.2) {
        // General poly with interaction:
        //Poly4:log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*
        r = std::exp(theta(0) * (s/tau) * (s/tau) + theta(1) * (s/tau)
                         + theta(2) * (s/tau) * (t/tau));

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_scalar()");
    }

    return r;
}



/*
 compute_r_vec: elementwise r(s_i, t_i; theta, sce) for vectors s and t (same length).

 Let s* = s/tau and t* = t/tau with tau = 20.0 (fixed scaling).

 Scenarios:
 2.1 (Sigmoid):
 r = 1 / (1 + exp( -theta1 · (s* − 0.5 · t*) ))

 2.2 (Polynomial in s only):
 log r = theta1 · s* + theta2 · (s*)^2     // no t dependence

 1.1 (Bell / bump with shift):
 log r = −theta1 · (s* − 0.5 · t*)^2 + θ2 · s*

 1.2 (General polynomial with interaction):
 log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*

 Returns:
 arma::vec of length s.n_elem.
 */
// [[Rcpp::export]]
arma::vec compute_r_vec(arma::vec s,
                        arma::vec t,
                        arma::vec theta,
                        double sce)
{
    arma::vec r(s.n_elem);
    double tau = 20.0;

    if (sce == 2.1) {
        // Sigmoid shape centered near s* = 0.5 · t*
        r = 1.0 / (1.0 + exp(-theta(0) * ((s/tau) - 0.5 * (t/tau))));

    } else if (sce == 2.2) {
        // Polynomial in s only (no t):
        //log r = theta1 · s* + theta2 · (s*)^2
        r = exp(theta(0) * (s/tau) + theta(1) * (s/tau) % (s/tau));

    } else if (sce == 1.1) {
        // Bell:
        //log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        arma::vec diff =(s/tau) - 0.5 * (t/tau);
        r = exp(-theta(0) * diff % diff + theta(1) * (s/tau));

    } else if (sce == 1.2) {
        // General poly with interaction:
        //log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*
        r = exp(theta(0) * (s/tau) % (s/tau) + theta(1) * (s/tau)
                         + theta(2) * (s/tau) % (t/tau));

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_vec()");
    }

    return r;
}


/*
 compute_r_dr: vectorized r(s_i, t_i; theta, sce) and its gradient wrt theta.

 Inputs:
 s, t   : vectors of the same length n (paired evaluations)
 theta  : parameter vector (length depends on scenario)
 sce    : scenario code in {1.1, 1.2, 2.1, 2.2}

 Scaling:
 Let s* = s/tau and t* = t/tau with tau = 20.0.

 Output:
 List with
 r  : arma::vec (n)          — r(s_i, t_i; theta, sce)
 dr : arma::mat (n × p)      — gradient wrt theta; column k is dr/dtheta_k

 Scenarios:
 2.1 (Sigmoid):
 r = 1 / (1 + exp( -theta1 · (s* − 0.5 · t*) ))
 dr/dtheta1 = r (1 − r) · u

 2.2 (Polynomial in s only):
 log r = theta1 · s* + theta2 · (s*)^2     // no t dependence
 dr/dtheta1 = r · s*
 dr/dtheta2 = r · (s*)^2

 1.1 (Bell / bump with shift):
 log r = −theta1 · (s* − 0.5 · t*)^2 + θ2 · s*
 dr/dtheta1 = r · (− diff^2)
 dr/dtheta2 = r · s*

 1.2 (General polynomial with s–t interaction):
 log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*
 dr/dtheta1 = r · (s*)^2
 dr/dtheta2 = r · s*
 dr/dtheta3 = r · s* · t*
 */
// [[Rcpp::export]]
Rcpp::List compute_r_dr(arma::vec s,
                        arma::vec t,
                        arma::vec theta,
                        double sce)
{
    double tau = 20.0;
    arma::vec r(s.n_elem);
    arma::mat dr(s.n_elem, theta.n_elem, arma::fill::zeros);

    if (sce == 2.1) {
        // Sigmoid shape centered near s* = 0.5 · t*
        r = 1.0 / (1.0 + exp(-theta(0) * ((s/tau) - 0.5 * (t/tau))));
        dr.col(0) = r % (1 - r) % ((s/tau) - 0.5 * (t/tau));

    } else if (sce == 2.2) {
        // Polynomial in s only (no t):
        //Poly2:log r = theta1 · s* + theta2 · (s*)^2
        r = exp(theta(0) * (s/tau) + theta(1) * (s/tau) % (s/tau));
        dr.col(0) = r % (s/tau);
        dr.col(1) = r % (s/tau) % (s/tau);

    } else if (sce == 1.1) {
        // Bell:
        //Poly3:log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        arma::vec diff =(s/tau) - 0.5 * (t/tau);
        r = exp(-theta(0) * diff % diff + theta(1) * (s/tau));
        dr.col(0) = -r % diff % diff;
        dr.col(1) = r % (s/tau);

    } else if (sce == 1.2) {
        // General poly with interaction:
        //Poly4:log r = theta1 · (s*)^2 + theta2 · s* + theta3 · s* · t*
        r = exp(theta(0) * (s/tau) % (s/tau) + theta(1) * (s/tau)
                    + theta(2) * (s/tau) % (t/tau));
        dr.col(0) = r % (s/tau) % (s/tau);
        dr.col(1) = r % (s/tau);
        dr.col(2) = r % (s/tau) % (t/tau);

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_dr()");
    }

    return Rcpp::List::create(
        Rcpp::Named("r") = r,
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
