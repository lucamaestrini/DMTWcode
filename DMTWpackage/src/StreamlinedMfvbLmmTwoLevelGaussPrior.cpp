#include <RcppArmadillo.h>
#include "SolveTwoLevelSparseMatrix.h"
using namespace Rcpp;
using namespace arma;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List StreamlinedMfvbLmmTwoLevelGaussPrior(const arma::field<arma::vec> yvecField,
                                                const arma::field<arma::mat> XmatField,
                                                const arma::field<arma::mat> ZmatField,
                                                const arma::vec muBetaRandA,
                                                const arma::mat SigmaBetaRandA,
                                                const double nusigma2,
                                                const double ssigma2,
                                                const double nuSigma,
                                                const arma::vec sSigma,
                                                const int pR,
                                                const int pA,
                                                const int pS,
                                                const int maxIter,
                                                bool verbose) {

  // yvecField: field of response variable vectors, for all i
  // XmatField: field of fixed effects design matrices, for all i
  // ZmatField: field of random effects design matrices, for all i
  // muBetaRandA: hyperparameter mean vector for (beta^R, beta^A)
  // SigmaBetaRandA: hyperparameter covariance matrix for (beta^R, beta^A)
  // nusigma2: degrees of freedom hyperparameter for sigma2
  // ssigma2: scale hyperparameter for sigma2
  // nuSigma: degrees of freedom hyperparameters for Sigma
  // sSigma: vector of scale hyperparameters for Sigma
  // pR: number of fixed effects associated to random effects
  // pA: number of additional fixed effects
  // pS: number of fixed effects subject to Gaussian prior
  // maxIter: number of iterations over which to run the algorithm
  // verbose: boolean, whether for printing the evolution of iterations

  // extract model dimensions:

  const int m = yvecField.n_elem;
  const int q = ZmatField(1).n_cols;
  const int p = pR + pA + pS;

  // organize fields taken in input from vector-form to matrix-form
  // and recreate the oVec vector:

  arma::field<arma::vec> y(m);
  arma::field<arma::mat> X(m);
  arma::field<arma::mat> Z(m);

  arma::field<arma::mat> XTX(m);
  arma::field<arma::mat> ZTZ(m);
  arma::field<arma::mat> XTZ(m);
  arma::field<arma::mat> ZTy(m);
  arma::mat SUMXTX(p, p, fill::zeros);
  arma::vec SUMXTy(p, fill::zeros);

  arma::vec oVec(m, fill::zeros);
  arma::vec otildeVec(m, fill::zeros);
  int h = 0;
  for(int i = 0; i < m; i++){
    y(i) = yvecField(h);
    X(i) = XmatField(h);
    Z(i) = ZmatField(h);

    XTX(i) = trans(X(i)) * X(i);
    ZTZ(i) = trans(Z(i)) * Z(i);
    XTZ(i) = trans(X(i)) * Z(i);
    ZTy(i) = trans(Z(i)) * y(i);

    oVec(i) = y(i).n_elem;
    otildeVec(i) = oVec(i) + p + q;

    SUMXTX = SUMXTX + trans(X(i)) * X(i);
    SUMXTy = SUMXTy + trans(X(i)) * y(i);
    h++;
  }

  // initializing variational parameters:

  arma::vec x1(p, fill::zeros);
  arma::mat Asup11(p, p, fill::zeros);
  arma::field<arma::vec> x2i(m);
  arma::field<arma::mat> Asup22i(m);
  arma::field<arma::mat> Asup12i(m);
  arma::mat Lambda_q_Sigma(q, q, fill::zeros);
  arma::mat Lambda_q_ASigma(q, q, fill::zeros);
  arma::mat M_q_recip_Sigma(q, q, fill::eye);
  arma::mat M_q_recip_ASigma(q, q, fill::eye);
  double lambda_q_sigsq = 0;
  double mu_q_recip_sigsq = 1.0;
  double lambda_q_a_sigsq = 0;
  double mu_q_recip_a_sigsq = 1.0;

  // useful stuff for computations:

  arma::mat invSigmabetastar(p, p, fill::zeros);
  invSigmabetastar(span(0, p-1), span(0, p-1)) = inv_sympd(SigmaBetaRandA);
  arma::vec mubetastar(p, fill::zeros);
  mubetastar(span(0, p-1)) = solve(SigmaBetaRandA, muBetaRandA);

  arma::vec a1(p, fill::zeros);
  arma::mat Ainf11(p, p, fill::zeros);
  arma::field<arma::vec> a2i(m);
  arma::field<arma::mat> Ainf22i(m);
  arma::field<arma::mat> Ainf12i(m);

  for(int i = 0; i < m; i++){
    a2i(i) = zeros<vec>(q);             x2i(i) = zeros<vec>(q);
    Ainf22i(i) = zeros<mat>(q, q);      Asup22i(i) = zeros<mat>(q, q);
    Ainf12i(i) = zeros<mat>(p, q);      Asup12i(i) = zeros<mat>(p, q);
  }

  ///////////////////
  // Fixed updates //
  ///////////////////

  const double xi_q_sigsq = accu(oVec) + nusigma2;
  const double xi_q_a_sigsq = nusigma2 + 1.0;
  const double xi_q_Sigma = nuSigma + m + 2.0 * q - 2.0;
  const double xi_q_ASigma = nuSigma + q;

  bool converged = false;
  int itNum = 0;

  while(!converged){

    itNum++;

    // Updating a1, Ainf11, Ainf22i, Ainf12i, a2i

    a1 = mu_q_recip_sigsq * SUMXTy + mubetastar;
    Ainf11 = mu_q_recip_sigsq * SUMXTX + invSigmabetastar;

    for(int i = 0; i < m; i++){
      a2i(i) = mu_q_recip_sigsq * ZTy(i);
      Ainf22i(i) = mu_q_recip_sigsq * ZTZ(i) + M_q_recip_Sigma;
      Ainf12i(i) = mu_q_recip_sigsq * XTZ(i);
      a2i(i) = mu_q_recip_sigsq * ZTy(i);
    }

    /////////////////////////////
    // Updates for q*(beta, u) //
    // being a N(mu, Sigma)    //
    /////////////////////////////

    SolveTwoLevelSparseMatrix(a1, Ainf11, a2i, Ainf22i, Ainf12i,
                              x1, Asup11, x2i, Asup22i, Asup12i,
                              otildeVec);

    //////////////////////////////////
    // Updates for q*(sigsq)        //
    // being an Inv-X^2(xi, lambda) //
    /////////////////////////////////

    // also compute Lambda_q_Sigma

    lambda_q_sigsq = mu_q_recip_a_sigsq;
    Lambda_q_Sigma = M_q_recip_ASigma;
    for(int i = 0; i < m; i++){
      Lambda_q_Sigma = Lambda_q_Sigma + (x2i(i) * trans(x2i(i))) + Asup22i(i);
      lambda_q_sigsq = lambda_q_sigsq +
        accu(pow(y(i) - X(i) * x1 - Z(i) * x2i(i), 2.0)) +
        trace(XTX(i) * Asup11) +
        trace(ZTZ(i) * Asup22i(i)) +
        2.0 * trace(trans(XTZ(i)) * Asup12i(i));
    }

    mu_q_recip_sigsq = xi_q_sigsq/lambda_q_sigsq;

    //////////////////////////////////
    // Updates for q*(a_sigsq)      //
    // being an Inv-X^2(xi, lambda) //
    /////////////////////////////////

    lambda_q_a_sigsq = mu_q_recip_sigsq + 1.0/(nusigma2 * pow(ssigma2, 2.0));
    mu_q_recip_a_sigsq = xi_q_a_sigsq/lambda_q_a_sigsq;

    ////////////////////////////////////////////////////
    // Updates for q*(Sigma)                          //
    // being an Inverse-G-Wishart(G_full, xi, Lambda) //
    ////////////////////////////////////////////////////

    M_q_recip_Sigma = (xi_q_Sigma - q + 1.0) * inv(Lambda_q_Sigma);

    ////////////////////////////////////////////////////
    // Updates for q*(ASigma)                         //
    // being an Inverse-G-Wishart(G_diag, xi, Lambda) //
    ////////////////////////////////////////////////////

    Lambda_q_ASigma = diagmat(M_q_recip_Sigma) + diagmat(1.0/nuSigma * pow(sSigma, -2.0));
    M_q_recip_ASigma = xi_q_ASigma * inv(Lambda_q_ASigma);

    if(verbose) Rcout << "Iteration number " << itNum << "\n";
    if(itNum >= maxIter){
      converged = true;
      if(verbose) Rcout << "Maximum number of MFVB iterations reached." << "\n";
    }

  }

  return Rcpp::List::create(Rcpp::Named("q_beta_params")=Rcpp::List::create(Rcpp::Named("mu_q_beta")=x1,
                                                                            Rcpp::Named("Sigma_q_beta")=Asup11),
                            Rcpp::Named("q_u_params")=Rcpp::List::create(Rcpp::Named("mu_q_u")=x2i,
                                                                         Rcpp::Named("Sigma_q_u")=Asup22i),
                            Rcpp::Named("q_sigsq_params")=Rcpp::List::create(Rcpp::Named("xi_q_sigsq")=xi_q_sigsq,
                                                                             Rcpp::Named("lambda_q_sigsq")=lambda_q_sigsq),
                            Rcpp::Named("q_Sigma_params")=Rcpp::List::create(Rcpp::Named("xi_q_Sigma")=xi_q_Sigma,
                                                                             Rcpp::Named("Lambda_q_Sigma")=Lambda_q_Sigma));
}
