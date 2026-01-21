#ifndef MU_SURV_H
#define MU_SURV_H

#include <RcppArmadillo.h>

// Survivor mean estimator (full version: checks Z >= tau inside)
double mu_surv(double s,
               double h,
               const arma::vec& beta,
               const arma::mat& X,
               const arma::vec& Y,
               const arma::vec& S,
               const arma::vec& Z,
               double tau);

// Survivor mean estimator (core version: assumes you already filtered Z >= tau)
double mu_surv_core(double s,
                    double h,
                    const arma::vec& beta,
                    const arma::mat& X,
                    const arma::vec& Y,
                    const arma::vec& S);

#endif
