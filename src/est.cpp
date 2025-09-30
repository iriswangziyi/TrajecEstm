// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "est.h"
#include "utils.h"

// Compute the PPL assuming semi-parametric baseline trajectory but no survival model

// --------------------
// PPL (Sigmoid Version)
// --------------------
// [[Rcpp::export]]
double PPL_sigmoid(arma::vec btj,
                   arma::uword j,
                    const arma::mat& X,
                    const arma::vec& Y,
                    const arma::vec& S,
                    const arma::vec& Z,
                    const arma::mat& Kmat,
                    double h1,
                    double tau0,
                    double tau1) {

    // Number of observation
    int n = S.n_elem;
    int p = X.n_rows;

    // Split parameter vector
    arma::vec bj = btj(arma::regspace<arma::uvec>(0, p-1));  // β
    double theta1 = exp(btj(p));                             // θ
    // double theta2 = exp(btj(p+1));

    // Compute reusable vectors
    arma::vec xbj = X.t() * bj;                // eta = Xβ
    arma::vec exp_xbj = exp(xbj);             // precompute exp(Xβ)
    arma::vec r = 1 / (1 + exp(theta1 * (S - Z / 2))); // r_j(S, Z; θ)

    arma::vec eXr = exp_xbj % r; // Precompute before loop

    // Pre-compute kernel weight matrix
    //arma::mat Kmat = matK(Z, h1);    // n_j x n_j matrix
    //arma::mat Kmat = matK_dispatch(Z, h1, use_sparse);

    double logPPL = 0;
    // Loop over individuals for partial log-likelihood
    for( int i = 0; i<n; ++i ) {
        if(Z(i) >= tau0 && Z(i) <= tau1) {
            //double den = arma::dot(Kmat.row(i), exp_xbj % r); //takes long time
            //double den = arma::dot(Kmat.row(i), eXr);
            double den = arma::dot(Kmat.col(i), eXr);
            logPPL +=  Y(i) *  ( xbj(i) + log(r(i)) - log(den) );
        }
    }

    return -logPPL;
}


//--------------------------------------------------
// Objective functoion: joint PPL value + gradient
//--------------------------------------------------
struct PPLObjective_V1 {
    const arma::uword  j;          // event‑type index (unused internally for now)
    const arma::mat&   X;          // p × n design matrix (columns = subjects)
    const arma::vec&   Y_S;        // counting‑process increment
    const arma::vec&   S;          // visit time s
    const arma::vec&   Z;          // age since baseline t
    const arma::mat&   Kmat;       // symmetric kernel weight matrix
    const double       tau0;       // lower boundary for t
    const double       tau1;       // upper boundary for t
    const double       sce;         // Scenario

    PPLObjective_V1(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_S_,
                    const arma::vec& S_,  const arma::vec& Z_,  const arma::mat& Kmat_,
                    double tau0_,  double tau1_, double sce_)
        : j(j_), X(X_), Y_S(Y_S_), S(S_), Z(Z_), Kmat(Kmat_),
          tau0(tau0_), tau1(tau1_), sce(sce_) {}

    // --------------------------------------------------
    // Evaluate −logPPL and its gradient in one pass
    // --------------------------------------------------
    double EvaluateWithGradient(const arma::mat& btj,
                                arma::mat& grad) {
        const int n = S.n_elem;
        const int p = X.n_rows;
        const int q = btj.n_elem - p;

        // Split parameter vector
        arma::vec beta  = btj.rows(0, p - 1);   // first p rows → β
        //p to the end of btj is theta
        // theta now is not converted to exp(theta)

        arma::vec theta = btj.rows(p, p+q-1);   // first p rows → β

        //std::exp(btj(p, 0));  // last row (log θ)

        // Pre‑compute shared quantities
        arma::vec xbj     = X.t() * beta;                        // XiT B
        arma::vec exp_xbj = arma::exp(xbj);

        //TODO 06/20
        //arma::vec r       = 1.0 / (1.0 + arma::exp(theta * (S - Z * 0.5)));
        //arma::vec dr      = -r % (1.0 - r) % (S - Z * 0.5);      // dr/dtheta

        //change to this
        Rcpp::List r_dr = compute_r_dr(S, Z, theta, sce);          // externally computed
        //arma::mat dr = compute_dr(S, Z, theta);            // externally computed
        arma::vec r = r_dr["r"];
        arma::mat dr = r_dr["dr"]; //nxq
        arma::mat dr_t = dr.t(); //qxn
        arma::vec eXr = exp_xbj % r;  // exp(η) · r

        // Accumulators
        double    logPPL      = 0.0;
        arma::vec grad_beta(p, arma::fill::zeros);
        arma::vec grad_theta(q, arma::fill::zeros);

        // Main loop over subjects
        for (int i = 0; i < n; ++i) {
            if (Z(i) < tau0 || Z(i) > tau1) continue;   // boundary skip

            //double denom = arma::dot(Kmat.col(i), eXr); // Kᵢ·(exp_xbj ∘ r)
            double yi    = Y_S(i);

            // ---- gradient contributions ----
            arma::vec ki = Kmat.col(i);
            arma::vec kexp = ki % exp_xbj;

            arma::vec weight = kexp % r;            // length n
            double denom = arma::sum(weight);

            //arma::vec weighted_X     = X * weight;  // p‑vector
            //arma::vec grad_i_beta    = X.col(i) - X * weight / denom;

            //double weighted_dlogr = arma::sum(kexp % dr) / denom;
            //double grad_i_theta   = dr(i) / r(i) - (arma::sum(kexp % dr) / denom);

            grad_beta  += yi * (X.col(i) - X * weight / denom);
            grad_theta += yi * (dr_t.col(i) / r(i) - (dr_t*kexp / denom));

            // Objective increment
            logPPL += yi * ( xbj(i) + std::log(r(i)) - std::log(denom) );
        }

        // Assemble gradient in the same (p+1) × 1 matrix shape
        grad.zeros(btj.n_rows, btj.n_cols);
        grad.rows(0, p - 1) = -grad_beta;
        grad.rows(p, p+q-1)  = -grad_theta;  // chain rule for log θ

        return -logPPL;
    }
};

//--------------------------------------------------
// R‑level wrapper
//--------------------------------------------------
// [[Rcpp::export]]
arma::vec estimate_beta_theta_lbfgs_V1(arma::uword          j,
                                       const arma::mat&    X,
                                       const arma::vec&    Y_S,
                                       const arma::vec&    S,
                                       const arma::vec&    Z,
                                       const arma::mat&    Kmat,
                                       double              tau0,
                                       double              tau1,
                                       double              sce,
                                       arma::vec           init,
                                       double              tol       = 1e-8,
                                       std::size_t         max_iter  = 1000) {

    PPLObjective_V1 fn(j, X, Y_S, S, Z, Kmat, tau0, tau1, sce);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);   // solution written into `init`
    return init;              //(beta hat, theta hat)
}


//ANOTHER WAY!!!
//Yifei cool code updated ver
//--------------------------------------------------
// Objective function: joint PPL value + gradient
//--------------------------------------------------
struct PPLObjective_V2 {
    const arma::uword  j;          // event‑type index (unused internally for now)
    const arma::mat&   X;          // p × n design matrix (columns = subjects)
    const arma::vec&   Y_S;        // counting‑process increment
    const arma::vec&   S;          // visit time s
    const arma::vec&   Z;          // age since baseline t
    const arma::mat&   Kmat;       // symmetric kernel weight matrix
    const double       tau0;       // lower boundary for t
    const double       tau1;       // upper boundary for t
    const double       sce;         // Scenario

    PPLObjective_V2(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_S_,
                    const arma::vec& S_,  const arma::vec& Z_,  const arma::mat& Kmat_,
                    double tau0_,  double tau1_, double sce_)
        : j(j_), X(X_), Y_S(Y_S_), S(S_), Z(Z_), Kmat(Kmat_),
          tau0(tau0_), tau1(tau1_), sce(sce_) {}

    // --------------------------------------------------
    // Evaluate −logPPL and its gradient in one pass
    // --------------------------------------------------

    //NEW USE THIS written by Fei
    double EvaluateWithGradient(const arma::mat& btj,
                                arma::mat& grad) {
        //const arma::uword n = S.n_elem;
        const arma::uword p = X.n_rows;
        const arma::uword q = btj.n_rows - p;  // number of theta parameters

        // Extract parameters
        const arma::vec beta = btj.rows(0, p - 1);
        const arma::vec theta = btj.rows(p, p + q - 1);

        // Precompute shared quantities
        arma::vec xbj = X.t() * beta;                      // linear predictor
        arma::vec exp_xbj = arma::exp(xbj);                // exp(eta)

        Rcpp::List r_dr = compute_r_dr(S, Z, theta, sce);              // externally computed
        //arma::mat dr = compute_dr(S, Z, theta);            // externally computed
        arma::vec r = r_dr["r"];
        arma::mat dr = r_dr["dr"];

        // Combined indicator and Y_S vector
        arma::vec Y_valid = Y_S % ((Z >= tau0) % (Z <= tau1));

        // Precompute terms
        arma::vec exp_xbj_r = exp_xbj % r;                 // (n by 1)
        arma::mat weight_mat = Kmat.each_col() % exp_xbj_r; // (n by n)

        arma::vec denom = arma::sum(weight_mat, 0).t();    // (n by 1)


        arma::vec grad_beta = X * (Y_valid - weight_mat * (Y_valid / denom));  // (p by 1)

        arma::mat weight_mat_theta = Kmat.each_col() % exp_xbj; // (n by n), no r here
        arma::vec grad_theta = dr.t() * (Y_valid / r - weight_mat_theta * (Y_valid / denom));  // (q by 1)

        // Compute logPPL
        double logPPL = arma::dot(Y_valid, xbj + arma::log(r) - arma::log(denom));

        // Assemble gradient
        grad.zeros(btj.n_rows, btj.n_cols);
        grad.rows(0, p - 1) = -grad_beta;
        grad.rows(p, p + q - 1) = -grad_theta;

        return -logPPL;
    }

};



//--------------------------------------------------
// yifei R‑level wrapper
//--------------------------------------------------
// [[Rcpp::export]]
arma::vec estimate_beta_theta_lbfgs_V2(arma::uword          j,
                                       const arma::mat&    X,
                                       const arma::vec&    Y_S,
                                       const arma::vec&    S,
                                       const arma::vec&    Z,
                                       const arma::mat&    Kmat,
                                       double              tau0,
                                       double              tau1,
                                       double              sce,
                                       arma::vec           init,
                                       double              tol       = 1e-8,
                                       std::size_t         max_iter  = 1000) {

    PPLObjective_V2 fn(j, X, Y_S, S, Z, Kmat, tau0, tau1, sce);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);   // solution written into `init`
    return init;              // (beta hat, theta hat)
}

