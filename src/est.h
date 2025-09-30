#ifndef EST_H
#define EST_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

// Declare PPL_sigmoid
double PPL_sigmoid(arma::vec btj,
                   arma::uword j,
                   const arma::mat& X,
                   const arma::vec& Y_S,
                   const arma::vec& S,
                   const arma::vec& Z,
                   const arma::mat& Kmat,
                   double h1,
                   double tau0,
                   double tau1);

//--------------------------------------------------
// under Objective function: joint PPL value + gradient
//--------------------------------------------------
arma::vec estimate_beta_theta_lbfgs_V1(arma::uword          j,
                                       const arma::mat&    X,
                                       const arma::vec&    Y_S,
                                       const arma::vec&    S,
                                       const arma::vec&    Z,
                                       const arma::mat&    Kmat,
                                       double              tau0,
                                       double              tau1,
                                       double              sce,
                                       arma::vec           init,
                                       double              tol,
                                       std::size_t         max_iter);

arma::vec estimate_beta_theta_lbfgs_V2(arma::uword          j,
                                       const arma::mat&    X,
                                       const arma::vec&    Y_S,
                                       const arma::vec&    S,
                                       const arma::vec&    Z,
                                       const arma::mat&    Kmat,
                                       double              tau0,
                                       double              tau1,
                                       double              sce,
                                       arma::vec           init,
                                       double              tol,
                                       std::size_t         max_iter);

#endif
