#include "objective.h"
#include "utils.h"

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

PPLObjective::PPLObjective(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_A_,
             const arma::vec& A_,  const arma::vec& Z_,  const arma::mat& Kmat_,
             double tau0_,  double tau1_)
    : j(j_), X(X_), Y_A(Y_A_), A(A_), Z(Z_), Kmat(Kmat_),
      tau0(tau0_), tau1(tau1_) {}

// --------------------------------------------------
// Evaluate −logPPL and its gradient in one pass
// --------------------------------------------------
double PPLObjective::EvaluateWithGradient(const arma::mat& btj,
                            arma::mat& grad) {
    const int n = A.n_elem;
    const int p = X.n_rows;

    // Split parameter vector
    arma::vec beta  = btj.rows(0, p - 1);   // first p rows → β
    double    theta = std::exp(btj(p, 0));  // last row (log θ)

    // Pre‑compute shared quantities
    arma::vec xbj     = X.t() * beta;                        // η_i = Xᵢᵗ β
    arma::vec exp_xbj = arma::exp(xbj);
    arma::vec r       = 1.0 / (1.0 + arma::exp(theta * (A - Z * 0.5)));
    arma::vec dr      = -r % (1.0 - r) % (A - Z * 0.5);      // ∂r/∂θ
    //arma::vec eXr     = exp_xbj % r;                         // exp(η) · r

    // Accumulators
    double    logPPL      = 0.0;
    arma::vec grad_beta(p, arma::fill::zeros);
    double    grad_theta  = 0.0;

    // Main loop over subjects
    for (int i = 0; i < n; ++i) {
        if (Z(i) < tau0 || Z(i) > tau1) continue;   // boundary skip

        //double denom = arma::dot(Kmat.col(i), eXr); // Kᵢ·(exp_xbj ∘ r)
        double yi    = Y_A(i);

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
        grad_theta += yi * (dr(i) / r(i) - (arma::sum(kexp % dr) / denom));

        // Objective increment
        logPPL += yi * ( xbj(i) + std::log(r(i)) - std::log(denom) );
    }

    // Assemble gradient in the same (p+1) × 1 matrix shape
    grad.zeros(btj.n_rows, btj.n_cols);
    grad.rows(0, p - 1) = -grad_beta;
    grad(p, 0)         = -grad_theta * theta;  // chain rule for log θ

    return -logPPL;
    }
}
