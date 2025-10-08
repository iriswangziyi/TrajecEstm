// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include <algorithm>   // std::max, std::min
#include <cmath>       // std::fabs
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

    } else if (sce == 1.2) {
        // Bell:
        //Poly3:log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        double diff = (s/tau) - 0.5 * (t/tau);
        r = std::exp(-theta(0) * diff * diff + theta(1) * (s/tau));

    } else if (sce == 1.1) {
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

    } else if (sce == 1.2) {
        // Bell:
        //log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        arma::vec diff =(s/tau) - 0.5 * (t/tau);
        r = exp(-theta(0) * diff % diff + theta(1) * (s/tau));

    } else if (sce == 1.1) {
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

 1.2 (Bell / bump with shift):
 log r = −theta1 · (s* − 0.5 · t*)^2 + θ2 · s*
 dr/dtheta1 = r · (− diff^2)
 dr/dtheta2 = r · s*

 1.1 (General polynomial with s–t interaction):
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

    } else if (sce == 1.2) {
        // Bell:
        //Poly3:log r = −theta1 · (s* − 0.5 · t*)^2 + theta2 · s*
        arma::vec diff =(s/tau) - 0.5 * (t/tau);
        r = exp(-theta(0) * diff % diff + theta(1) * (s/tau));
        dr.col(0) = -r % diff % diff;
        dr.col(1) = r % (s/tau);

    } else if (sce == 1.1) {
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

//10-08

// --- helpers (explicit std/arma) --------------------------------------------
static inline arma::vec clamp_vec(const arma::vec& x, double lo, double hi) {
    arma::vec y = x;
    y.transform([&](double v){ return std::max(lo, std::min(hi, v)); });
    return y;
}
static inline bool sce_is(double sce, double target) {
    return std::fabs(sce - target) < 1e-9;
}
static inline double med(const arma::vec& v) {
    if (v.n_elem == 0) return 0.0;
    return arma::median(v);
}

/*
 compute_r_vec: elementwise r(s_i, t_i; theta, sce) for vectors s and t.

 Scaling: s* = s/τ, t* = t/τ, Δ* = (t - s)/τ with τ = 20.
 Centering (3.X only): s̃ = s* - m_s, t̃ = t* - m_t, Δ̃ = Δ* - m_Δ (medians on scaled axes).
 For log-scale models we clamp η = log r to [-20, 20] before exp().

 Scenarios (legacy kept):
 2.1 Sigmoid:         r = 1 / (1 + exp( -θ1 (s* − 0.5 t*) ))
 2.2 s-only poly:     log r = θ1 s* + θ2 (s*)^2
 1.2 Bell:            log r = −θ1 (s* − 0.5 t*)^2 + θ2 s*
 1.1 General poly:    log r = θ1 (s*)^2 + θ2 s* + θ3 s* t*

 New centered family:
 3.1 s-only poly:     log r = θ1 s̃ + θ2 s̃^2
 3.2 curvature in s:  log r = θ1 s̃^2
 3.3 Δ poly:          log r = θ1 Δ̃ + θ2 Δ̃^2
 3.4 s × Δ:           log r = θ1 s̃ + θ2 Δ̃ + θ3 s̃ Δ̃
 3.5 bump near onset: log r = θ1 − θ2 Δ̃^2 + θ3 s̃
 3.6 s × t:           log r = θ1 s̃ + θ2 t̃ + θ3 s̃ t̃
 */
// [[Rcpp::export]]
arma::vec compute_r_vec2(const arma::vec& s,
                        const arma::vec& t,
                        const arma::vec& theta,
                        double sce)
{
    if (s.n_elem != t.n_elem) Rcpp::stop("s and t must have same length");

    const double tau = 20.0;

    // scaled axes
    arma::vec sp = s / tau;             // s*
    arma::vec tp = t / tau;             // t*
    arma::vec dp = (t - s) / tau;       // Δ*

    // centers (medians on scaled axes)
    const double ms = med(sp);
    const double mt = med(tp);
    const double md = med(dp);

    // centered axes
    arma::vec st = sp - ms;             // s̃
    arma::vec tt = tp - mt;             // t̃
    arma::vec dt = dp - md;             // Δ̃

    arma::vec r(s.n_elem, arma::fill::zeros);

    // ---- legacy (uncentered) -------------------------------------------------
    if (sce_is(sce, 2.1)) {
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 2.1");
        r = 1.0 / (1.0 + arma::exp(-theta(0) * (sp - 0.5 * tp)));

    } else if (sce_is(sce, 2.2)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 2.2");
        arma::vec eta = theta(0) * sp + theta(1) * (sp % sp);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 1.2)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 1.2");
        arma::vec diff = sp - 0.5 * tp;
        arma::vec eta  = -theta(0) * (diff % diff) + theta(1) * sp;
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 1.1)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 1.1");
        arma::vec eta = theta(0) * (sp % sp) + theta(1) * sp + theta(2) * (sp % tp);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

        // ---- new centered 3.X family --------------------------------------------
    } else if (sce_is(sce, 3.1)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.1");
        arma::vec eta = theta(0) * st + theta(1) * (st % st);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.2)) {
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 3.2");
        arma::vec eta = theta(0) * (st % st);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.3)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.3");
        arma::vec eta = theta(0) * dt + theta(1) * (dt % dt);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.4)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.4");
        arma::vec eta = theta(0) * st + theta(1) * dt + theta(2) * (st % dt);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.5)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.5");
        arma::vec eta = theta(0) - theta(1) * (dt % dt) + theta(2) * st; // expect theta(1) > 0
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.6)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.6");
        arma::vec eta = theta(0) * st + theta(1) * tt + theta(2) * (st % tt);
        r = arma::exp(clamp_vec(eta, -20.0, 20.0));

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_vec()");
    }

    return r;
}

// ---- add this tiny scalar helper ONCE (used by compute_r_scalar) ----
static inline double clamp_double(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// Optional: fixed centers for scalar evaluation (debug/predict path)
static constexpr double MS0 = 0.5;   // m_s on scaled axis
static constexpr double MT0 = 0.5;   // m_t on scaled axis
static constexpr double MD0 = 0.25;  // m_Δ on scaled axis

// ===================================================================
// [[Rcpp::export]]
double compute_r_scalar2(double s,
                        double t,
                        const arma::vec& theta,
                        double sce)
{
    const double tau = 20.0;

    // scaled axes
    const double sp = s / tau;          // s*
    const double tp = t / tau;          // t*
    const double dp = (t - s) / tau;    // Δ*

    // centered (fixed reference for scalar variant)
    const double st = sp - MS0;         // s̃
    const double tt = tp - MT0;         // t̃
    const double dt = dp - MD0;         // Δ̃

    double r = 0.0;

    // ---- legacy (uncentered) ---------------------------------------
    if (sce_is(sce, 2.1)) {
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 2.1");
        const double z = sp - 0.5 * tp;
        r = 1.0 / (1.0 + std::exp(-theta(0) * z));

    } else if (sce_is(sce, 2.2)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 2.2");
        double eta = theta(0) * sp + theta(1) * (sp * sp);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 1.2)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 1.2");
        const double diff = sp - 0.5 * tp;
        double eta = -theta(0) * (diff * diff) + theta(1) * sp;
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 1.1)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 1.1");
        double eta = theta(0) * (sp * sp) + theta(1) * sp + theta(2) * (sp * tp);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

        // ---- centered 3.X family ---------------------------------------
    } else if (sce_is(sce, 3.1)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.1");
        double eta = theta(0) * st + theta(1) * (st * st);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.2)) {
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 3.2");
        double eta = theta(0) * (st * st);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.3)) {
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.3");
        double eta = theta(0) * dt + theta(1) * (dt * dt);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.4)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.4");
        double eta = theta(0) * st + theta(1) * dt + theta(2) * (st * dt);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.5)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.5");
        double eta = theta(0) - theta(1) * (dt * dt) + theta(2) * st; // expect theta(1) > 0
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else if (sce_is(sce, 3.6)) {
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.6");
        double eta = theta(0) * st + theta(1) * tt + theta(2) * (st * tt);
        r = std::exp(clamp_double(eta, -20.0, 20.0));

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_scalar()");
    }

    return r;
}



/*
 compute_r_dr: vectorized r(s_i, t_i; theta, sce) and its gradient wrt theta.

 Inputs:
 s, t   : length-n vectors (paired)
 theta  : parameter vector (length depends on scenario)
 sce    : scenario code {1.1, 1.2, 2.1, 2.2, 3.1–3.6}

 Scaling:
 s* = s/τ, t* = t/τ, Δ* = (t - s)/τ  with τ = 20.
 Centering (3.X only):
 s̃ = s* - median(s*),  t̃ = t* - median(t*),  Δ̃ = Δ* - median(Δ*).

 Output (R list):
 r  : arma::vec (n)
 dr : arma::mat (n × p), column k is ∂r/∂θ_k

 Notes on clamping:
 For log-scale models we compute η (unclamped), then r = exp( clamp(η, -20, 20) ).
 The gradient is r * ∂η/∂θ on rows where η ∈ (-20, 20), and 0 on rows that clip.
 */
// (helpers clamp_vec, sce_is, med are assumed to be defined ONCE above)


// [[Rcpp::export]]
Rcpp::List compute_r_dr2(const arma::vec& s,
                        const arma::vec& t,
                        const arma::vec& theta,
                        double sce)
{
    if (s.n_elem != t.n_elem) Rcpp::stop("s and t must have same length");

    const arma::uword n = s.n_elem;
    const double tau = 20.0;

    // scaled axes
    arma::vec sp = s / tau;            // s*
    arma::vec tp = t / tau;            // t*
    arma::vec dp = (t - s) / tau;      // Δ*

    // centers (medians on scaled axes)
    const double ms = med(sp);
    const double mt = med(tp);
    const double md = med(dp);

    // centered axes
    arma::vec st = sp - ms;            // s̃
    arma::vec tt = tp - mt;            // t̃
    arma::vec dt = dp - md;            // Δ̃

    arma::vec r(n, arma::fill::zeros);
    arma::mat dr(n, theta.n_elem, arma::fill::zeros);

    // ---------------- legacy (uncentered) --------------------------------------
    if (sce_is(sce, 2.1)) {
        // Sigmoid: r = 1 / (1 + exp(-θ1 * u)), with u = s* - 0.5 t*
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 2.1");
        arma::vec u = sp - 0.5 * tp;
        r = 1.0 / (1.0 + arma::exp(-theta(0) * u));
        dr.col(0) = r % (1.0 - r) % u;

    } else if (sce_is(sce, 2.2)) {
        // log r = θ1 s* + θ2 (s*)^2
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 2.2");
        arma::vec eta = theta(0) * sp + theta(1) * (sp % sp);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        // active rows where clamp is inactive
        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r;                // w = r on active rows; 0 otherwise
        w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % sp;
        dr.col(1) = w % (sp % sp);

    } else if (sce_is(sce, 1.2)) {
        // log r = -θ1 (s* - 0.5 t*)^2 + θ2 s*
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 1.2");
        arma::vec diff = sp - 0.5 * tp;
        arma::vec eta  = -theta(0) * (diff % diff) + theta(1) * sp;
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % (-diff % diff);
        dr.col(1) = w % sp;

    } else if (sce_is(sce, 1.1)) {
        // log r = θ1 (s*)^2 + θ2 s* + θ3 s* t*
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 1.1");
        arma::vec eta = theta(0) * (sp % sp) + theta(1) * sp + theta(2) * (sp % tp);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % (sp % sp);
        dr.col(1) = w % sp;
        dr.col(2) = w % (sp % tp);

        // ---------------- centered 3.X family --------------------------------------
    } else if (sce_is(sce, 3.1)) {
        // log r = θ1 s̃ + θ2 s̃^2
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.1");
        arma::vec eta = theta(0) * st + theta(1) * (st % st);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % st;
        dr.col(1) = w % (st % st);

    } else if (sce_is(sce, 3.2)) {
        // log r = θ1 s̃^2
        if (theta.n_elem != 1u) Rcpp::stop("theta length must be 1 for sce 3.2");
        arma::vec eta = theta(0) * (st % st);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % (st % st);

    } else if (sce_is(sce, 3.3)) {
        // log r = θ1 Δ̃ + θ2 Δ̃^2
        if (theta.n_elem != 2u) Rcpp::stop("theta length must be 2 for sce 3.3");
        arma::vec eta = theta(0) * dt + theta(1) * (dt % dt);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % dt;
        dr.col(1) = w % (dt % dt);

    } else if (sce_is(sce, 3.4)) {
        // log r = θ1 s̃ + θ2 Δ̃ + θ3 s̃Δ̃
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.4");
        arma::vec eta = theta(0) * st + theta(1) * dt + theta(2) * (st % dt);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % st;
        dr.col(1) = w % dt;
        dr.col(2) = w % (st % dt);

    } else if (sce_is(sce, 3.5)) {
        // log r = θ1 − θ2 Δ̃^2 + θ3 s̃   (expect θ2 > 0)
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.5");
        arma::vec eta = theta(0) - theta(1) * (dt % dt) + theta(2) * st;
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w;                      // ∂η/∂θ1 = 1
        dr.col(1) = w % (-(dt % dt));       // ∂η/∂θ2 = -Δ̃^2
        dr.col(2) = w % st;                 // ∂η/∂θ3 = s̃

    } else if (sce_is(sce, 3.6)) {
        // log r = θ1 s̃ + θ2 t̃ + θ3 s̃ t̃
        if (theta.n_elem != 3u) Rcpp::stop("theta length must be 3 for sce 3.6");
        arma::vec eta = theta(0) * st + theta(1) * tt + theta(2) * (st % tt);
        arma::vec eta_c = clamp_vec(eta, -20.0, 20.0);
        r = arma::exp(eta_c);

        arma::uvec active_u = (eta > -20.0) % (eta < 20.0);
        arma::vec  w = r; w.elem(arma::find(active_u == 0)).zeros();

        dr.col(0) = w % st;
        dr.col(1) = w % tt;
        dr.col(2) = w % (st % tt);

    } else {
        Rcpp::stop("Unsupported sce value in compute_r_dr()");
    }

    return Rcpp::List::create(
        Rcpp::Named("r")  = r,
        Rcpp::Named("dr") = dr
    );
}

