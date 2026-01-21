// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//--------------------------------------------------
// Objective: survivor model
//   Maximize log L(β) for long-term survivors
//--------------------------------------------------
struct SurvivorObjective {
    const arma::mat& X;      // p × n (columns = subjects)
    const arma::vec& Y;      // n × 1, marker at baseline Y_i(A_i)
    const arma::vec& A;      // n × 1, baseline time A_i
    const arma::mat& Kmat;   // n × n, K_h(A_i - A_k)
    const double tau0;       // lower bound for baseline time
    const double tau;        // τ (long-term survivor time threshold)

    SurvivorObjective(const arma::mat&  X_,
                      const arma::vec&  Y_,
                      const arma::vec&  A_,
                      const arma::mat&  Kmat_,
                      double tau0_,
                      double tau_)
        : X(X_), Y(Y_), A(A_), Kmat(Kmat_),
          tau0(tau0_), tau(tau_) {}

    // Evaluate −logL and its gradient in one pass
    double EvaluateWithGradient(const arma::mat& beta_mat,
                                arma::mat& grad)
    {
        const arma::uword p = X.n_rows;
        //const arma::uword n = X.n_cols;

        // β is p×1 (same shape as beta_mat)
        arma::vec beta = beta_mat.rows(0, p - 1);

        // Linear predictor and exp(η)
        arma::vec xb     = X.t() * beta;    // n × 1
        arma::vec exp_xb = arma::exp(xb);   // n × 1

        // Indicators:
        //  - subject is event-free at τ (took this to the wrapper)
        //  - baseline time A in [tau0, tau]
        //TODO: change A to S (measurement time) in notation
        //arma::vec atRiskZ = arma::conv_to<arma::vec>::from(Z >= tau);      // 0/1
        arma::vec inBandA = arma::conv_to<arma::vec>::from((A >= tau0) % (A <= tau));

        // I(A-window) * Y_i(A_i)
        arma::vec Y_valid = Y % inBandA;   // n × 1

        // For risk set sums, only keep subjects with Z_k ≥ τ
        //arma::vec exp_xb_at = exp_xb % atRiskZ;      // n × 1

        // K_weight(i, k) = Kmat(i, k) * exp_xb_at(k)
        arma::mat K_weight = Kmat;                   // n × n
        K_weight.each_row() %= exp_xb.t();

        // denom_i = ∑_k I(Z_k ≥ τ) K_h(A_i - A_k) exp(β^T X_k)
        arma::vec denom = arma::sum(K_weight, 1);    // row sums, n × 1

        // Guard against zero denominators: if denom_i == 0,
        //   - set denom_i = 1 to avoid log(0)
        //   - force Y_valid_i = 0 so it contributes nothing
        arma::uvec zero_idx = arma::find(denom <= 0);
        if (zero_idx.n_elem > 0) {
            denom.elem(zero_idx).ones();
            Y_valid.elem(zero_idx).zeros();
            K_weight.rows(zero_idx).zeros();
        }

        // W(i, k) = K_weight(i, k) / denom_i
        arma::mat W = K_weight;                      // n × n
        W.each_row() /= denom.t();

        // weighted_X is p × n, column i = ∑_k W(i, k) X_k
        arma::mat weighted_X = X * W.t();

        // Gradient of logL wrt β:
        //   ∑_i Y_valid_i [X_i − ∑_k w_ik X_k]
        arma::vec grad_beta = X * Y_valid - weighted_X * Y_valid;  // p × 1

        // Log-likelihood
        arma::vec log_denom = arma::log(denom);
        double logL = arma::dot(Y_valid, xb - log_denom);

        // Return negative log-likelihood and gradient
        grad.zeros(beta_mat.n_rows, beta_mat.n_cols);
        grad.rows(0, p - 1) = -grad_beta;

        return -logL;
    }
};

//--------------------------------------------------
// R-level wrapper
//--------------------------------------------------
// [[Rcpp::export]]
arma::vec estimate_beta_survivor_lbfgs(const arma::mat& X,
                                       const arma::vec& Y,
                                       const arma::vec& A,
                                       const arma::mat& Kmat,
                                       double           tau0,
                                       double           tau,
                                       arma::vec        init,
                                       double           tol      = 1e-8,
                                       std::size_t      max_iter = 1000)
{
    SurvivorObjective fn(X, Y, A, Kmat, tau0, tau);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);   // solution written into `init`
    return init;              // β̂
}
