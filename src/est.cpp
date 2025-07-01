// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu.h"
#include "utils.h"
#include "objective.h"

// Compute the PPL assuming semi-parametric baseline trajectory but no survival model

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

