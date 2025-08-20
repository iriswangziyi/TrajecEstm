// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "utils.h"

//--------------------------------------------------
// Objective function: joint PPL value + gradient
//--------------------------------------------------
struct PPLObjective_NP {
    const arma::uword  j;          // event‑type index (unused internally for now)
    const arma::mat&   X;          // p × n design matrix (columns = subjects)
    const arma::vec&   Y_A;        // counting‑process increment
    const arma::vec&   A;          // visit time s
    const arma::vec&   Z;          // age since baseline t
    const arma::mat&   Kmat;       // symmetric kernel weight matrix
    const double       d;        // adjustment constant
    const double       tau0;       // lower boundary for t
    const double       tau1;       // upper boundary for t

    PPLObjective_NP(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_A_,
                    const arma::vec& A_,  const arma::vec& Z_,  const arma::mat& Kmat_,
                    double d_, double tau0_,  double tau1_)
        : j(j_), X(X_), Y_A(Y_A_), A(A_), Z(Z_), Kmat(Kmat_),
          d(d_), tau0(tau0_), tau1(tau1_){}

    // --------------------------------------------------
    // Evaluate −logPPL and its gradient in one pass
    // --------------------------------------------------

    double EvaluateWithGradient(const arma::mat& bj,
                                arma::mat& grad) {
        //const arma::uword n = A.n_elem;
        const arma::uword p = X.n_rows;

        // Extract parameters
        const arma::vec beta = bj.rows(0, p - 1);
        //const arma::vec theta = btj.rows(p, p + q - 1);

        // Pre-compute shared quantities
        arma::vec xbj = X.t() * beta;                      // linear predictor
        arma::vec exp_xbj = arma::exp(xbj);                // exp(eta)

        // Combined indicator and Y_A vector
        arma::vec Y_valid = Y_A % (A >= tau0) % ((Z - d >= A) % (Z <= tau1));

        // Precompute terms
        arma::mat weight_mat = Kmat.each_col() % exp_xbj; // (n by n)
        arma::vec denom = arma::sum(weight_mat, 0).t();    // (n by 1)

        arma::vec grad_beta = X * (Y_valid - weight_mat * (Y_valid / denom));  // (p by 1)

        // Compute logPPL
        const double eps = 1e-6;            // floor
        arma::vec denom_safe = denom;        // if need the original later
        denom_safe.elem( arma::find(denom_safe <= eps) ).fill(eps);

        double logPPL = arma::dot(Y_valid, xbj - arma::log(denom_safe));

        // Assemble gradient
        grad.zeros(bj.n_rows, bj.n_cols);
        grad.rows(0, p - 1) = -grad_beta;

        return -logPPL;
    }

};

//--------------------------------------------------
// R‑level wrapper for est_NP
//--------------------------------------------------
// [[Rcpp::export]]
arma::vec estimate_beta_NP(arma::uword          j,
                           const arma::mat&    X,
                           const arma::vec&    Y_A,
                           const arma::vec&    A,
                           const arma::vec&    Z,
                           const arma::mat&    Kmat,
                           double              d,
                           double              tau0,
                           double              tau1,
                           arma::vec           init,
                           double              tol       = 1e-8,
                           std::size_t         max_iter  = 1000) {

    PPLObjective_NP fn(j, X, Y_A, A, Z, Kmat, d, tau0, tau1);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);   // solution written into `init`
    return init;              // (beta hat)
}

