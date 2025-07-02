#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

// Kernel matrix functions
arma::mat matK(const arma::vec& Z, double h1);
arma::sp_mat matK_sparse(const arma::vec& Z, double h1);
arma::mat matK_dispatch(const arma::vec& Z, double h1, Rcpp::LogicalVector use_sparse);

// Compute both r and dr as vectors
Rcpp::List compute_r_dr(arma::vec a,
                        arma::vec z,
                        arma::vec theta,
                        double sce);

// Compute only r as a vector
arma::vec compute_r_vec(arma::vec a,
                        arma::vec z,
                        arma::vec theta,
                        double sce);

// Compute r and dr for a single (a, z) pair
Rcpp::List compute_r_dr_scalar(double a,
                               double z,
                               arma::vec theta,
                               double sce);

// Compute only r for a single (a, z) pair
double compute_r_scalar(double a,
                        double z,
                        arma::vec theta,
                        double sce);


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

double rfun2(double a,
             double t,
             arma::vec theta);

#endif
