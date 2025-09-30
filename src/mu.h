#ifndef MU_H
#define MU_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//--------------------------------------------------
// TO be filled
//--------------------------------------------------
double mu_r(arma::uword j,
            double t,
            double s,
            double h,
            const arma::vec& btj,
            const arma::mat& X,
            const arma::vec& Y,
            const arma::uvec& delPi,
            const arma::vec& S,
            const arma::vec& Z,
            double sce);

#endif
