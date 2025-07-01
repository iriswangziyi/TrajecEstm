#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <RcppArmadillo.h>


// Struct for PPL objective + gradient

//--------------------------------------------------
// Objective functor: joint PPL value + gradient
//--------------------------------------------------
struct PPLObjective {
    const arma::uword  j;          // event‑type index (unused internally for now)
    const arma::mat&   X;          // p × n design matrix (columns = subjects)
    const arma::vec&   Y_A;        // counting‑process increment
    const arma::vec&   A;          // visit time s
    const arma::vec&   Z;          // age since baseline t
    const arma::mat&   Kmat;       // symmetric kernel weight matrix
    const double       tau0;       // lower boundary for t
    const double       tau1;       // upper boundary for t
    
    PPLObjective(arma::uword  j_,  const arma::mat&  X_,  const arma::vec& Y_A_,
                 const arma::vec& A_,  const arma::vec& Z_,  const arma::mat& Kmat_,
                 double tau0_,  double tau1_);
    
    // --------------------------------------------------
    // Evaluate −logPPL and its gradient in one pass
    // --------------------------------------------------
    double EvaluateWithGradient(const arma::mat& btj,
                                arma::mat& grad);
};

#endif