#ifndef MU_H
#define MU_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//--------------------------------------------------
// TO be filled
//--------------------------------------------------

double mu_NP(arma::uword j,
             double t,
             double a,
             double h1,
             double h2,
             const arma::vec& bj,
             const arma::mat& X,
             const arma::vec& Y,
             const arma::uvec& delPi,
             const arma::vec& A,
             const arma::vec& Z);

#endif
