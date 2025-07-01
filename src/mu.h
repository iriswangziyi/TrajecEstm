#ifndef MU_H
#define MU_H

#include <RcppArmadillo.h>

// Declare mu6_gamma
double mu6_gamma(arma::uword j,
                 double t,
                 double a,
                 double h,
                 const arma::vec& btj,
                 const arma::mat& X,
                 const arma::vec& Y,
                 const arma::uvec& delPi,
                 const arma::vec& A,
                 const arma::vec& Z);

double mu6_exp(arma::uword j,
               double t,
               double a,
               double h,
               const arma::vec& btj,
               const arma::mat& X,
               const arma::vec& Y,
               const arma::uvec& delPi,
               const arma::vec& A,
               const arma::vec& Z);

double mu6_sigmoid(arma::uword j,
                   double t,
                   double a,
                   double h,
                   const arma::vec& btj,
                   const arma::mat& X,
                   const arma::vec& Y,
                   const arma::uvec& delPi,
                   const arma::vec& A,
                   const arma::vec& Z);

double mu6_r(arma::uword j,
             double t,
             double a,
             double h,
             const arma::vec& btj,
             const arma::mat& X,
             const arma::vec& Y,
             const arma::uvec& delPi,
             const arma::vec& A,
             const arma::vec& Z);

#endif
