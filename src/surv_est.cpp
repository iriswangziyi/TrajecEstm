// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//--------------------------------------------------
// Objective: survivor model
//   Maximize log L(β) for long-term survivors
// Note: S is the measurement time (baseline case is allowed)
//--------------------------------------------------
struct SurvivorObjective {
    const arma::mat& X;      // p × n (columns = subjects)
    const arma::vec& Y;      // n × 1, marker (aligned with S usage in objective)
    const arma::vec& S;      // n × 1, measurement/baseline time
    const arma::mat& Kmat;   // n × n, K_h(S_i - S_k)
    const double tau0;       // lower bound for S
    const double tau1;       // upper bound for S

    SurvivorObjective(const arma::mat&  X_,
                      const arma::vec&  Y_,
                      const arma::vec&  S_,
                      const arma::mat&  Kmat_,
                      double tau0_,
                      double tau1_)
        : X(X_), Y(Y_), S(S_), Kmat(Kmat_),
          tau0(tau0_), tau1(tau1_) {}

    double EvaluateWithGradient(const arma::mat& beta_mat,
                                arma::mat& grad)
    {
        const arma::uword p = X.n_rows;

        arma::vec beta = beta_mat.rows(0, p - 1);

        arma::vec xb     = X.t() * beta;    // n × 1
        arma::vec exp_xb = arma::exp(xb);   // n × 1

        // Domain restriction for boundary issue: S in [tau0, tau1]
        arma::vec inBandS = arma::conv_to<arma::vec>::from((S >= tau0) % (S <= tau1));

        // Only i's in the band contribute via Y_valid
        arma::vec Y_valid = Y % inBandS;

        // K_weight(i, k) = Kmat(i, k) * exp_xb(k)
        arma::mat K_weight = Kmat;
        K_weight.each_row() %= exp_xb.t();

        arma::vec denom = arma::sum(K_weight, 1);    // n × 1

        // Guard against zero denominators
        arma::uvec zero_idx = arma::find(denom <= 0);
        if (zero_idx.n_elem > 0) {
            denom.elem(zero_idx).ones();
            Y_valid.elem(zero_idx).zeros();
            K_weight.rows(zero_idx).zeros();
        }

        arma::mat W = K_weight;
        W.each_row() /= denom.t();

        arma::mat weighted_X = X * W.t();

        arma::vec grad_beta = X * Y_valid - weighted_X * Y_valid;

        arma::vec log_denom = arma::log(denom);
        double logL = arma::dot(Y_valid, xb - log_denom);

        grad.zeros(beta_mat.n_rows, beta_mat.n_cols);
        grad.rows(0, p - 1) = -grad_beta;

        return -logL;
    }
};

// [[Rcpp::export]]
arma::vec estimate_beta_survivor_lbfgs(const arma::mat& X,
                                       const arma::vec& Y,
                                       const arma::vec& S,
                                       const arma::mat& Kmat,
                                       double           tau0,
                                       double           tau1,
                                       arma::vec        init,
                                       double           tol      = 1e-8,
                                       std::size_t      max_iter = 1000)
{
    SurvivorObjective fn(X, Y, S, Kmat, tau0, tau1);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);
    return init;
}
