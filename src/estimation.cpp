// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppEnsmallen.h>

// Compute the PPL assuming semi-parametric baseline trajectory but no survival model


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
// PPL (Sigmoid Version)
// --------------------
// [[Rcpp::export]]
double PPL_sigmoid(arma::vec btj,
                   arma::uword j,
                    const arma::mat& X,
                    const arma::vec& Y_A,
                    const arma::vec& A,
                    const arma::vec& Z,
                    const arma::mat& Kmat,
                    double h1,
                    double tau0,
                    double tau1) {

    // Number of observation
    int n = A.n_elem;
    int p = X.n_rows;

    // Split parameter vector
    arma::vec bj = btj(arma::regspace<arma::uvec>(0, p-1));  // β
    double theta1 = exp(btj(p));                             // θ
    // double theta2 = exp(btj(p+1));

    // Compute reusable vectors
    arma::vec xbj = X.t() * bj;                // eta = Xβ
    arma::vec exp_xbj = exp(xbj);             // precompute exp(Xβ)
    arma::vec r = 1 / (1 + exp(theta1 * (A - Z / 2))); // r_j(A, Z; θ)

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
            logPPL +=  Y_A(i) *  ( xbj(i) + log(r(i)) - log(den) );
        }
    }

    return -logPPL;
}



//--------------------------------------------------
// Objective functor: joint PPL value + gradient
//--------------------------------------------------
struct PPLObjective {
    const arma::uword  j;          // event‑type index (unused internally for now)
    const arma::mat&   X;          // p × n design matrix (columns = subjects)
    const arma::vec&   Y_A;        // counting‑process increment
    const arma::vec&   A;          // visit time s
    const arma::vec&   Z;          // age since baseline t
    const arma::mat&   Kmat;       // symmetric kernel weight matrix
    const double       tau0;       // lower boundary for t
    const double       tau1;       // upper boundary for t

    PPLObjective(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_A_,
                 const arma::vec& A_,  const arma::vec& Z_,  const arma::mat& Kmat_,
                 double tau0_,  double tau1_)
        : j(j_), X(X_), Y_A(Y_A_), A(A_), Z(Z_), Kmat(Kmat_),
          tau0(tau0_), tau1(tau1_) {}

    // --------------------------------------------------
    // Evaluate −logPPL and its gradient in one pass
    // --------------------------------------------------
    double EvaluateWithGradient(const arma::mat& btj,
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
};

//--------------------------------------------------
// R‑level wrapper
//--------------------------------------------------
// [[Rcpp::export]]
arma::vec estimate_beta_theta_lbfgs(arma::uword          j,
                                    const arma::mat&    X,
                                    const arma::vec&    Y_A,
                                    const arma::vec&    A,
                                    const arma::vec&    Z,
                                    const arma::mat&    Kmat,
                                    double              tau0,
                                    double              tau1,
                                    arma::vec           init,
                                    double              tol       = 1e-8,
                                    std::size_t         max_iter  = 1000) {

    PPLObjective fn(j, X, Y_A, A, Z, Kmat, tau0, tau1);

    ens::L_BFGS opt;
    opt.MaxIterations()   = max_iter;
    opt.MinGradientNorm() = tol;

    opt.Optimize(fn, init);   // solution written into `init`
    return init;              // (β̂, log θ̂)
}


// [[Rcpp::export]]
double PPL6_gamma(arma::uword j,
                  arma::vec btj,
                  const arma::mat& X,
                  const arma::vec& Y_A,
                  const arma::vec& A,
                  const arma::vec& Z,
                  const arma::uvec& delPi,
                  double h1,
                  double tau0,
                  double tau1) {

    // Number of observations
    int n = A.n_elem;
    double logPPL = 0;
    int p = X.n_rows;

    Rcpp::Function dg("dgamma"); //use R function in CPP
    Rcpp::Function pg("pgamma");

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    double theta1 = exp(btj(p));

    Rcpp::NumericVector gA = pg(A, theta1);
    //Rcpp::NumericVector gZ = pg(Z, theta1);
    Rcpp::NumericVector r = gA;///gZ;
    //Rcpp::NumericVector gA = dg(A, theta(0), theta(1));
    //Rcpp::NumericVector r = gA +theta(2);
    arma::vec xbj = X.t() * bj;

    for( int i = 0; i<n; i++ )
    {
        if(delPi(i) == j && Z(i) >= tau0 && Z(i) <= tau1)
        {
            double den = 0;
            for( int k = 0; k<n; k++ )
            {
                if(delPi(k) == j &&  Z(i)-Z(k) < h1 && Z(i)-Z(k) > -h1)
                {
                    den = den +  0.75 * ( 1-pow(( Z(i)-Z(k) )/h1,2) ) / h1 * exp( xbj(k) ) * r[k];
                }
            }
            logPPL = logPPL +  Y_A(i) *  ( xbj(i) + log(r(i)) - log(den) );
        }
    }



    return -logPPL;
}

// [[Rcpp::export]]
double PPL6_exp(arma::uword j,
                arma::vec btj,
                const arma::mat& X,
                const arma::vec& Y_A,
                const arma::vec& A,
                const arma::vec& Z,
                const arma::uvec& delPi,
                double h1,
                double tau0,
                double tau1) {

    // Number of observationsd
    int n = A.n_elem;
    double logPPL = 0;
    int p = X.n_rows;


    Rcpp::Function pe("pexp");

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    double theta1 = exp(btj(p));

    Rcpp::NumericVector gA = pe(A, theta1);
    //Rcpp::NumericVector gZ = pg(Z, theta1);
    Rcpp::NumericVector r = gA;///gZ;
    //Rcpp::NumericVector gA = dg(A, theta(0), theta(1));
    //Rcpp::NumericVector r = gA +theta(2);
    arma::vec xbj = X.t() * bj;

    for( int i = 0; i<n; i++ )
    {
        if(delPi(i) == j && Z(i) >= tau0 && Z(i) <= tau1)
        {
            double den = 0;
            for( int k = 0; k<n; k++ )
            {
                if(delPi(k) == j &&  Z(i)-Z(k) < h1 && Z(i)-Z(k) > -h1)
                {
                    den = den +  0.75 * ( 1-pow(( Z(i)-Z(k) )/h1,2) ) / h1 * exp( xbj(k) ) * r[k];
                }
            }
            logPPL = logPPL +  Y_A(i) *  ( xbj(i) + log(r(i)) - log(den) );
        }
    }



    return -logPPL;
}

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



// [[Rcpp::export]]
double PPL6_r(arma::uword j,
              arma::vec btj,
              const arma::mat& X,
              const arma::vec& Y_A,
              const arma::vec& A,
              const arma::vec& Z,
              const arma::uvec& delPi,
              double h1) {

    // Number of observationsd
    int n = A.n_elem;
    double logPPL = 0;
    arma::uword p = X.n_rows;

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    arma::vec theta = (btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem-1)));
    // double theta2 = exp(btj(p+1));

    arma::vec r =  rfun(A,Z,theta);

    arma::vec xbj = X.t() * bj;

    for( int i = 0; i<n; i++ )
    {
        if(delPi(i) == j)
        {
            double den = 0;
            for( int k = 0; k<n; k++ )
            {
                if(delPi(k) == j && Z(i)-Z(k) < h1 && Z(i)-Z(k) > -h1)
                {
                    den = den +  0.75 * ( 1-pow(( Z(i)-Z(k) )/h1,2) ) / h1 * exp( xbj(k) ) * r(k);
                }
            }
            logPPL = logPPL +  Y_A(i) *  ( xbj(i) + log(r(i)) - log(den) );
        }
    }


    return -logPPL;
}



// [[Rcpp::export]]
double mu6_gamma(arma::uword j,
                 double t,
                 double a,
                 double h,
                 const arma::vec& btj,
                 const arma::mat& X,
                 const arma::vec& Y,
                 const arma::uvec& delPi,
                 const arma::vec& A,
                 const arma::vec& Z) {


    int n = A.n_elem;
    int p = X.n_rows;

    Rcpp::Function dg("dgamma");
    Rcpp::Function pg("pgamma");

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    arma::vec theta = exp(btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem-1)));

    Rcpp::NumericVector gA = pg(A, theta(0));
    Rcpp::NumericVector gZ = pg(Z, theta(0));
    Rcpp::NumericVector r = gA;
    //Rcpp::NumericVector gA = dg(A, theta(0), theta(1));
    //Rcpp::NumericVector r = gA +theta(2);



    arma::vec xbj = X.t() * bj;

    //
    //   double gj = 0;
    //
    //
    //   for (int i=0; i<n; i++) {
    //     if (delPi(i)==j) {
    //       double den2 = 0;
    //       for (int k=0; k<n; k++) {
    //         if(delPi(k) == j)
    //         {
    //           den2 = den2 + ( 0.75 * std::max( 1-pow((Z(k)-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(k) ) * r[k];
    //         }
    //       }
    //       gj = gj + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i)) / den2;
    //     }
    //   }
    //
    double num = 0;
    double den = 0;
    for (int i=0; i<n; i++) {
        if (delPi(i)==j) {
            den = den + ( 0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(i) ) * r[i];
            num = num + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i));
        }
    }



    Rcpp::NumericVector da = pg(a, theta(0));
    Rcpp::NumericVector pt = pg(t, theta(0));
    Rcpp::NumericVector mu6 = da * num / den;

    // Rcpp::NumericVector da = dg(a, theta(0), theta(1));
    // Rcpp::NumericVector mu6 = ( da+theta(2) ) * num / den;
    return mu6[0];
}

// [[Rcpp::export]]
double mu6_exp(arma::uword j,
               double t,
               double a,
               double h,
               const arma::vec& btj,
               const arma::mat& X,
               const arma::vec& Y,
               const arma::uvec& delPi,
               const arma::vec& A,
               const arma::vec& Z) {


    int n = A.n_elem;
    int p = X.n_rows;

    Rcpp::Function pe("pexp");

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    arma::vec theta = exp(btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem-1)));

    Rcpp::NumericVector gA = pe(A, theta(0));

    Rcpp::NumericVector r = gA;
    //Rcpp::NumericVector gA = dg(A, theta(0), theta(1));
    //Rcpp::NumericVector r = gA +theta(2);



    arma::vec xbj = X.t() * bj;

    //
    //   double gj = 0;
    //
    //
    //   for (int i=0; i<n; i++) {
    //     if (delPi(i)==j) {
    //       double den2 = 0;
    //       for (int k=0; k<n; k++) {
    //         if(delPi(k) == j)
    //         {
    //           den2 = den2 + ( 0.75 * std::max( 1-pow((Z(k)-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(k) ) * r[k];
    //         }
    //       }
    //       gj = gj + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i)) / den2;
    //     }
    //   }
    //
    double num = 0;
    double den = 0;
    for (int i=0; i<n; i++) {
        if (delPi(i)==j) {
            den = den + ( 0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(i) ) * r[i];
            num = num + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i));
        }
    }



    Rcpp::NumericVector da = pe(a, theta(0));
    Rcpp::NumericVector mu6 = da * num / den;

    // Rcpp::NumericVector da = dg(a, theta(0), theta(1));
    // Rcpp::NumericVector mu6 = ( da+theta(2) ) * num / den;
    return mu6[0];
}

//write a more generic mu


// [[Rcpp::export]]
double mu6_sigmoid(arma::uword j,
                   double t,
                   double a,
                   double h,
                   const arma::vec& btj,
                   const arma::mat& X,
                   const arma::vec& Y,
                   const arma::uvec& delPi,
                   const arma::vec& A,
                   const arma::vec& Z) {


    int n = A.n_elem;
    int p = X.n_rows;

    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    double theta1 = exp(btj(p));
    // double theta2 = exp(btj(p+1));

    //HERE IS r(s,t) related
    //TODO change to computer_r()
    //lucky no need to compute dr.
    arma::vec r =  1/(1+ exp(theta1*(A - Z/2)));

    arma::vec xbj = X.t() * bj;

    // double gj = 0;
    //
    //
    // for (int i=0; i<n; i++) {
    //   if (delPi(i)==j) {
    //     double den2 = 0;
    //     for (int k=0; k<n; k++) {
    //       if(delPi(k) == j)
    //       {
    //         den2 = den2 + ( 0.75 * std::max( 1-pow((Z(k)-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(k) ) * r[k];
    //       }
    //     }
    //     gj = gj + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i)) / den2;
    //   }
    // }

    double num = 0;
    double den = 0;
    for (int i=0; i<n; i++) {
        if (delPi(i)==j) {
            double w = 0.0;
            if( t < h )
            {
                w = 0.75 * std::max( 1-pow((h-Z(i))/h,2), 0.0 ) / h;
            }else{
                w = 0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h;
            }
            den = den + w * exp( xbj(i) ) * r[i];
            num = num + w * Y(i);
        }
    }

    return num/den/(1 + exp( theta1*(a-t/2) ));
}



// [[Rcpp::export]]
double mu6_r(arma::uword j,
             double t,
             double a,
             double h,
             const arma::vec& btj,
             const arma::mat& X,
             const arma::vec& Y,
             const arma::uvec& delPi,
             const arma::vec& A,
             const arma::vec& Z) {


    int n = A.n_elem;
    int p = X.n_rows;



    arma::vec bj = btj(arma::regspace<arma::uvec>(0,p-1));
    arma::vec theta = (btj(arma::regspace<arma::uvec>(p, 1, btj.n_elem-1)));

    arma::vec r =  rfun(A,Z,theta);


    arma::vec xbj = X.t() * bj;


    // double gj = 0;
    //
    //
    // for (int i=0; i<n; i++) {
    //   if (delPi(i)==j) {
    //     double den2 = 0;
    //     for (int k=0; k<n; k++) {
    //       if(delPi(k) == j)
    //       {
    //         den2 = den2 + ( 0.75 * std::max( 1-pow((Z(k)-Z(i))/h,2), 0.0 ) / h ) * exp( xbj(k) ) * r[k];
    //       }
    //     }
    //     gj = gj + (0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h * Y(i)) / den2;
    //   }
    // }

    double num = 0;
    double den = 0;
    for (int i=0; i<n; i++) {
        if (delPi(i)==j) {
            double w = 0.0;
            if( t < h )
            {
                w = 0.75 * std::max( 1-pow((h-Z(i))/h,2), 0.0 ) / h;
            }else{
                w = 0.75 * std::max( 1-pow((t-Z(i))/h,2), 0.0 ) / h;
            }
            den = den + w * exp( xbj(i) ) * r[i];
            num = num + w * Y(i);
        }
    }


    return num/den*rfun2(a,t,theta);
}


