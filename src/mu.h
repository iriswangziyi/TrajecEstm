#ifndef MU_H
#define MU_H

#include <RcppArmadillo.h>

//don’t need #include <RcppEnsmallen.h> (these functions don’t use it).
//Keeping it also ok, but slows compile a bit.

//--------------------------------------------------
// Model I mu estimators
//--------------------------------------------------

// Full version: handles delPi filtering internally
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

// Core version: assumes inputs already filtered to the desired group (e.g., delPi==j)
double mu_r_core(double t,
                 double s,
                 double h,
                 const arma::vec& btj,
                 const arma::mat& X,
                 const arma::vec& Y,
                 const arma::vec& S,
                 const arma::vec& Z,
                 double sce);

#endif
