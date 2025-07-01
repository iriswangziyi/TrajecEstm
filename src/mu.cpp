// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

#include "mu.h"
#include "utils.h"

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


