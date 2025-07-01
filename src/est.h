#ifndef EST_H
#define EST_H

#include <RcppArmadillo.h>

// Declare PPL_sigmoid
double PPL_sigmoid(arma::vec btj,
                   arma::uword j,
                   const arma::mat& X,
                   const arma::vec& Y_A,
                   const arma::vec& A,
                   const arma::vec& Z,
                   const arma::mat& Kmat,
                   double h1,
                   double tau0,
                   double tau1);

#endif
