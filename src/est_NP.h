#ifndef EST_H
#define EST_H

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>

//--------------------------------------------------
// TO be filled
//--------------------------------------------------
arma::vec estimate_beta_NP(arma::uword          j,
                           const arma::mat&    X,
                           const arma::vec&    Y_S,
                           const arma::vec&    S,
                           const arma::vec&    Z,
                           const arma::mat&    Kmat,
                           double              d,
                           double              tau0,
                           double              tau1,
                           arma::vec           init,
                           double              tol,
                           std::size_t         max_iter);


#endif
