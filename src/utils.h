#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

// Kernel matrix functions
arma::mat matK(const arma::vec& Z, double h1);
arma::sp_mat matK_sparse(const arma::vec& Z, double h1);
arma::mat matK_dispatch(const arma::vec& Z, double h1, Rcpp::LogicalVector use_sparse);

// Gradient function
arma::vec gradi(arma::vec btj,
                arma::uword j,
                const arma::mat& X,
                const arma::vec& Y_A,
                const arma::vec& A,
                const arma::vec& Z,
                const arma::mat& Kmat,
                double h1,
                double tau0,
                double tau1);

// rfun utilities
arma::vec rfun(arma::vec a, arma::vec t, arma::vec theta);
double rfun2(double a, double t, double theta);

#endif
