#ifndef MU_H
#define MU_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//might delete this gamma case in the future
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

//stick with this one
double mu_r(arma::uword j,
            double t,
            double a,
            double h,
            const arma::vec& btj,
            const arma::mat& X,
            const arma::vec& Y,
            const arma::uvec& delPi,
            const arma::vec& A,
            const arma::vec& Z,
            double sce);

#endif
