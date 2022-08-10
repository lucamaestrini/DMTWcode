#include <RcppArmadillo.h>
#include "SolveThreeLevelSparseMatrix.h"
#include "Lentz_Q.h"
#include "Lentz_R_nu.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List StreamlinedMfvbLmmThreeLevelGlobLocPrior(const arma::field<arma::vec> yvecField,
                                                    const arma::field<arma::mat> XmatField,
                                                    const arma::field<arma::mat> ZL1matField,
                                                    const arma::field<arma::mat> ZL2matField,
                                                    const arma::vec muBetaRandA,
                                                    const arma::mat SigmaBetaRandA,
                                                    const double nusigma2,
                                                    const double ssigma2,
                                                    const double stau2,
                                                    const double lambda,
                                                    const double nuSigmaL1,
                                                    const double nuSigmaL2,
                                                    const arma::vec sSigmaL1,
                                                    const arma::vec sSigmaL2,
                                                    const arma::vec nVec,
                                                    const int pR,
                                                    const int pA,
                                                    const int pS,
                                                    bool FactorizeZeta,
                                                    const std::string PriorType,
                                                    const int maxIter,
                                                    bool verbose) {

  // yvecField: field of response variable vectors, for all (i,j)
  // XmatField: field of fixed effects design matrices, for all (i,j)
  // ZL1matField: field of level-1 random effects design matrices, for all (i,j)
  // ZL2matField: field of level-2 random effects design matrices, for all (i,j)
  // muBetaRandA: hyperparameter mean vector for (beta^R, beta^A)
  // SigmaBetaRandA: hyperparameter covariance matrix for (beta^R, beta^A)
  // nusigma2: degrees of freedom hyperparameter for sigma2
  // ssigma2: scale hyperparameter for sigma2
  // stau2: scale hyperparameter for tau2
  // lambda: shape hyperparameter, only useful if PriorType=='Normal-Exponential-Gamma'
  // nuSigmaL1: degrees of freedom hyperparameters for SigmaL1
  // nuSigmaL2: degrees of freedom hyperparameters for SigmaL2
  // sSigmaL1: vector of scale hyperparameters for SigmaL1
  // sSigmaL2: vector of scale hyperparameters for SigmaL2
  // pR: number of fixed effects associated to random effects
  // pA: number of additional fixed effects
  // pS: number of fixed effects subject to global-local prior
  // FactorizeZeta: boolean, whether for considering q(zeta_h) q(a_{zeta_h}) (TRUE)
  //                or q(zeta_h) (FALSE) without introducing the augmented a_{zeta_h}
  // PriorType: string for the type of global local specification. one between
  //            'Laplace', 'Horseshoe' or 'Normal-Exponential-Gamma'.
  // maxIter: number of iterations over which to run the algorithm
  // verbose: boolean, whether for printing the evolution of iterations

  // extract model dimensions:

  const int m = nVec.n_elem;
  const int qL1 = ZL1matField(1).n_cols;
  const int qL2 = ZL2matField(1).n_cols;
  const int p = pR + pA + pS;

  const int maxn = max(nVec);

  // organize fields taken in input from vector-form to matrix-form
  // and recreate the oMat matrix:

  arma::field<arma::vec> y(m, maxn);
  arma::field<arma::mat> X(m, maxn);
  arma::field<arma::mat> ZL1(m, maxn);
  arma::field<arma::mat> ZL2(m, maxn);

  arma::field<arma::mat> XTX(m, maxn);
  arma::field<arma::mat> ZL1TZL1(m, maxn);
  arma::field<arma::mat> ZL2TZL2(m, maxn);
  arma::field<arma::mat> ZL1TX(m, maxn);
  arma::field<arma::mat> ZL2TX(m, maxn);
  arma::field<arma::mat> ZL1TZL2(m, maxn);
  arma::field<arma::vec> ZL2Ty(m, maxn);

  arma::mat SUMXTX(p, p, fill::zeros);
  arma::vec SUMXTy(p, fill::zeros);
  arma::field<arma::mat> SUMZL1TZL1i(m);
  arma::field<arma::mat> SUMXTZL1i(m);
  arma::field<arma::vec> SUMZL1Tyi(m);

  arma::mat oMat(m, maxn, fill::zeros);
  arma::mat otildeMat(m, maxn, fill::zeros);
  int h = 0;
  for(int i = 0; i < m; i++){
    SUMZL1TZL1i(i) = zeros<mat>(qL1, qL1);
    SUMXTZL1i(i) = zeros<mat>(p, qL1);
    SUMZL1Tyi(i) = zeros<vec>(qL1);
    for(int j = 0; j < nVec(i); j++){
      y(i,j) = yvecField(h);
      X(i,j) = XmatField(h);
      ZL1(i,j) = ZL1matField(h);
      ZL2(i,j) = ZL2matField(h);

      oMat(i,j) = y(i,j).n_elem;
      otildeMat(i,j) = oMat(i,j) + p + qL1 + qL2;

      XTX(i,j) = trans(X(i,j)) * X(i,j);
      ZL1TZL1(i,j) = trans(ZL1(i,j)) * ZL1(i,j);
      ZL2TZL2(i,j) = trans(ZL2(i,j)) * ZL2(i,j);
      ZL1TX(i,j) = trans(ZL1(i,j)) * X(i,j);
      ZL2TX(i,j) = trans(ZL2(i,j)) * X(i,j);
      ZL1TZL2(i,j) = trans(ZL1(i,j)) * ZL2(i,j);
      ZL2Ty(i,j) = trans(ZL2(i,j)) * y(i,j);

      SUMXTX = SUMXTX + XTX(i,j);
      SUMXTy = SUMXTy + trans(X(i,j)) * y(i,j);
      SUMZL1TZL1i(i) = SUMZL1TZL1i(i) + ZL1TZL1(i,j);
      SUMXTZL1i(i) = SUMXTZL1i(i) + trans(ZL1TX(i,j));
      SUMZL1Tyi(i) = SUMZL1Tyi(i) + trans(ZL1(i,j)) * y(i,j);

      h++;
    }
  }

  // initializing variational parameters:

  arma::vec x1(p, fill::zeros);
  arma::mat Asup11(p, p, fill::zeros);
  arma::field<arma::vec> x2i(m);
  arma::field<arma::mat> Asup22i(m);
  arma::field<arma::mat> Asup12i(m);
  arma::field<arma::vec> x2ij(m, maxn);
  arma::field<arma::mat> Asup22ij(m, maxn);
  arma::field<arma::mat> Asup12ij(m, maxn);
  arma::field<arma::mat> Asup12iandj(m, maxn);
  arma::vec mu_q_zeta = ones<vec>(pS);
  arma::vec mu_q_recip_zeta = ones<vec>(pS); // only useful for 'Normal-Exponential-Gamma' if FactorizeZeta=TRUE
  arma::vec mu_q_a_zeta = ones<vec>(pS); // useful for 'Horseshoe' if FactorizeZeta=TRUE
  arma::vec G = ones<vec>(pS);
  arma::mat Lambda_q_SigmaL1(qL1, qL1, fill::zeros);
  arma::mat Lambda_q_ASigmaL1(qL1, qL1, fill::zeros);
  arma::mat M_q_recip_SigmaL1(qL1, qL1, fill::eye);
  arma::mat M_q_recip_ASigmaL1(qL1, qL1, fill::eye);
  arma::mat Lambda_q_SigmaL2(qL2, qL2, fill::zeros);
  arma::mat Lambda_q_ASigmaL2(qL2, qL2, fill::zeros);
  arma::mat M_q_recip_SigmaL2(qL2, qL2, fill::eye);
  arma::mat M_q_recip_ASigmaL2(qL2, qL2, fill::eye);
  double lambda_q_sigsq = 0;
  double mu_q_recip_sigsq = 1.0;
  double lambda_q_a_sigsq = 0;
  double mu_q_recip_a_sigsq = 1.0;
  double lambda_q_tausq = 0;
  double mu_q_recip_tausq = 1.0;
  double lambda_q_a_tausq = 0;
  double mu_q_recip_a_tausq = 1.0;

  // useful stuff for computations:

  arma::vec mu_q_betaSpow2(pS, fill::zeros);

  arma::mat invSigmabetastar(p, p, fill::zeros);
  invSigmabetastar(span(0, (p-pS)-1), span(0, (p-pS)-1)) = inv_sympd(SigmaBetaRandA);
  arma::vec mubetastar(p, fill::zeros);
  mubetastar(span(0, (p-pS)-1)) = solve(SigmaBetaRandA, muBetaRandA);

  arma::vec a1(p, fill::zeros);
  arma::mat Ainf11(p, p, fill::zeros);
  arma::field<arma::vec> a2i(m);
  arma::field<arma::mat> Ainf22i(m);
  arma::field<arma::mat> Ainf12i(m);
  arma::field<arma::vec> a2ij(m, maxn);
  arma::field<arma::mat> Ainf22ij(m, maxn);
  arma::field<arma::mat> Ainf12ij(m, maxn);
  arma::field<arma::mat> Ainf12iandj(m, maxn);

  for(int i = 0; i < m; i++){
    a2i(i) = zeros<vec>(qL1);             x2i(i) = zeros<vec>(qL1);
    Ainf22i(i) = zeros<mat>(qL1, qL1);    Asup22i(i) = zeros<mat>(qL1, qL1);
    Ainf12i(i) = zeros<mat>(p, qL1);      Asup12i(i) = zeros<mat>(p, qL1);
    for(int j = 0; j < nVec(i); j++){
      a2ij(i,j) = zeros<vec>(qL2);              x2ij(i,j) = zeros<vec>(qL2);
      Ainf22ij(i,j) = zeros<mat>(qL2, qL2);     Asup22ij(i,j) = zeros<mat>(qL2, qL2);
      Ainf12ij(i,j) = zeros<mat>(p, qL2);       Asup12ij(i,j) = zeros<mat>(p, qL2);
      Ainf12iandj(i,j) = zeros<mat>(qL1, qL2);  Asup12iandj(i,j) = zeros<mat>(qL1, qL2);
    }
  }

  ///////////////////
  // Fixed updates //
  ///////////////////

  const double xi_q_sigsq = accu(oMat) + nusigma2;
  const double xi_q_a_sigsq = nusigma2 + 1.0;
  const double xi_q_tausq = pS + 1.0;
  const double xi_q_a_tausq = 2.0;
  const double xi_q_SigmaL1 = nuSigmaL1 + m + 2.0 * qL1 - 2.0;
  const double xi_q_SigmaL2 = nuSigmaL2 +  + accu(nVec) + 2.0 * qL2 - 2.0;
  const double xi_q_ASigmaL1 = nuSigmaL1 + qL1;
  const double xi_q_ASigmaL2 = nuSigmaL2 + qL2;

  bool converged = false;
  int itNum = 0;

  while(!converged){

    itNum++;

    // Updating a1, Ainf11, Ainf22i, Ainf12i, a2i, Ainf22ij, Ainf12iandj, Ainf12ij

    invSigmabetastar(span((p-pS), p-1), span((p-pS), p-1)).diag() = mu_q_recip_tausq * mu_q_zeta;

    a1 = mu_q_recip_sigsq * SUMXTy + mubetastar;
    Ainf11 = mu_q_recip_sigsq * SUMXTX + invSigmabetastar;

    for(int i = 0; i < m; i++){
      a2i(i) = mu_q_recip_sigsq * SUMZL1Tyi(i);
      Ainf22i(i) = mu_q_recip_sigsq * SUMZL1TZL1i(i) + M_q_recip_SigmaL1;
      Ainf12i(i) = mu_q_recip_sigsq * SUMXTZL1i(i);
      a2i(i) = mu_q_recip_sigsq * SUMZL1Tyi(i);
      for(int j = 0; j < nVec(i); j++){
        a2ij(i,j) = mu_q_recip_sigsq * ZL2Ty(i,j);
        Ainf22ij(i,j) = mu_q_recip_sigsq * ZL2TZL2(i,j) + M_q_recip_SigmaL2;
        Ainf12ij(i,j) = mu_q_recip_sigsq * trans(ZL2TX(i,j));
        Ainf12iandj(i,j) = mu_q_recip_sigsq * ZL1TZL2(i,j);
      }
    }

    /////////////////////////////
    // Updates for q*(beta, u) //
    // being a N(mu, Sigma)    //
    /////////////////////////////

    SolveThreeLevelSparseMatrix(a1, Ainf11, a2i, Ainf22i, Ainf12i, a2ij, Ainf22ij, Ainf12ij, Ainf12iandj,
                                x1, Asup11, x2i, Asup22i, Asup12i, x2ij, Asup22ij, Asup12ij, Asup12iandj,
                                otildeMat);

    // compute E_q[(beta^S)^2]:

    mu_q_betaSpow2 = pow(x1(span(pR+pA, p-1)), 2.0) + Asup11(span(pR+pA, p-1), span(pR+pA, p-1)).diag();

    //////////////////////////////////
    // Updates for q*(sigsq)        //
    // being an Inv-X^2(xi, lambda) //
    /////////////////////////////////

    // also compute Lambda_q_SigmaL1 and Lambda_q_SigmaL2

    lambda_q_sigsq = mu_q_recip_a_sigsq;
    Lambda_q_SigmaL1 = M_q_recip_ASigmaL1;
    Lambda_q_SigmaL2 = M_q_recip_ASigmaL2;
    for(int i = 0; i < m; i++){
      Lambda_q_SigmaL1 = Lambda_q_SigmaL1 + (x2i(i) * trans(x2i(i))) + Asup22i(i);
      for(int j = 0; j < nVec(i); j++){
        Lambda_q_SigmaL2 = Lambda_q_SigmaL2 + (x2ij(i,j) * trans(x2ij(i,j)) + Asup22ij(i,j));
        lambda_q_sigsq = lambda_q_sigsq +
          accu(pow(y(i,j) - X(i,j) * x1 - ZL1(i,j) * x2i(i) - ZL2(i,j) * x2ij(i,j), 2.0)) +
          trace(XTX(i,j) * Asup11) +
          trace(ZL1TZL1(i,j) * Asup22i(i)) +
          trace(ZL2TZL2(i,j) * Asup22ij(i,j)) +
          2.0 * trace(ZL1TX(i,j) * Asup12i(i)) +
          2.0 * trace(ZL2TX(i,j) * Asup12ij(i,j)) +
          2.0 * trace(ZL1TZL2(i,j) * Asup12iandj(i,j));
      }
    }

    mu_q_recip_sigsq = xi_q_sigsq/lambda_q_sigsq;

    //////////////////////////////////
    // Updates for q*(a_sigsq)      //
    // being an Inv-X^2(xi, lambda) //
    /////////////////////////////////

    lambda_q_a_sigsq = mu_q_recip_sigsq + 1.0/(nusigma2 * pow(ssigma2, 2.0));
    mu_q_recip_a_sigsq = xi_q_a_sigsq/lambda_q_a_sigsq;

    //////////////////////////////////
    // Updates for q*(tausq)        //
    // being an Inv-X^2(xi, lambda) //
    //////////////////////////////////

    lambda_q_tausq = mu_q_recip_a_tausq + accu(mu_q_zeta % mu_q_betaSpow2);
    mu_q_recip_tausq = xi_q_tausq/lambda_q_tausq;

    //////////////////////////////////
    // Updates for q*(a_tausq)      //
    // being an Inv-X^2(xi, lambda) //
    //////////////////////////////////

    lambda_q_a_tausq = mu_q_recip_tausq + pow(stau2, -2.0);
    mu_q_recip_a_tausq = xi_q_a_tausq/lambda_q_a_tausq;

    //////////////////////////////////////////////////////////////
    // Updates for q*(zeta), being                              //
    // Inverse-Gaussian if Laplace or Normal-Exponential-Gamma, //
    // Gamma if Horseshoe                                       //
    //                                                          //
    // and, if FactorizeZeta==TRUE                              //
    // updates for q*(a_zeta), being                            //
    // Gamma if Horseshoe or Normal-Exponential-Gamma,          //
    // useless if Laplace                                       //
    //////////////////////////////////////////////////////////////

    G = 0.5 * mu_q_recip_tausq * mu_q_betaSpow2;
    if(PriorType == "Laplace"){
      mu_q_zeta = sqrt(1.0/(2.0 * G));
    }
    if(PriorType == "Horseshoe"){
      if(FactorizeZeta){
        mu_q_zeta = 1.0/(mu_q_a_zeta + G);
        mu_q_a_zeta = 1.0/(mu_q_zeta + 1.0);
      } else {
        for(int h = 0; h < pS; h++){
          mu_q_zeta(h) = 1.0/(G(h) * Lentz_Q(G(h))) - 1.0;
        }
      }
    }
    if(PriorType == "Normal-Exponential-Gamma"){
      if(FactorizeZeta){
        mu_q_zeta = sqrt(mu_q_a_zeta/G);
        mu_q_recip_zeta = 1.0/mu_q_zeta + 1.0/(2.0 * mu_q_a_zeta);
        mu_q_a_zeta = (lambda + 1.0)/(mu_q_recip_zeta + 1.0);
      } else {
        for(int h = 0; h < pS; h++){
          mu_q_zeta(h) = (2.0 * lambda + 1.0) * Lentz_R_nu(sqrt(2.0 * G(h)), 2.0 * lambda) / sqrt(2.0 * G(h));
        }
      }
    }

    ////////////////////////////////////////////////////
    // Updates for q*(SigmaL1) and q*(SigmaL2)        //
    // being an Inverse-G-Wishart(G_full, xi, Lambda) //
    ////////////////////////////////////////////////////

    M_q_recip_SigmaL1 = (xi_q_SigmaL1 - qL1 + 1.0) * inv(Lambda_q_SigmaL1);
    M_q_recip_SigmaL2 = (xi_q_SigmaL2 - qL2 + 1.0) * inv(Lambda_q_SigmaL2);

    ////////////////////////////////////////////////////
    // Updates for q*(ASigmaL1)                       //
    // being an Inverse-G-Wishart(G_diag, xi, Lambda) //
    ////////////////////////////////////////////////////

    Lambda_q_ASigmaL1 = diagmat(M_q_recip_SigmaL1) + diagmat(1.0/nuSigmaL1 * pow(sSigmaL1, -2.0));
    M_q_recip_ASigmaL1 = xi_q_ASigmaL1 * inv(Lambda_q_ASigmaL1);

    ////////////////////////////////////////////////////
    // Updates for q*(ASigmaL2)                       //
    // being an Inverse-G-Wishart(G_diag, xi, Lambda) //
    ////////////////////////////////////////////////////

    Lambda_q_ASigmaL2 = diagmat(M_q_recip_SigmaL2) + diagmat(1.0/nuSigmaL2 * pow(sSigmaL2, -2.0));
    M_q_recip_ASigmaL2 = xi_q_ASigmaL2 * inv(Lambda_q_ASigmaL2);

    if(verbose) Rcout << "Iteration number " << itNum << "\n";
    if(itNum >= maxIter){
      converged = true;
      if(verbose) Rcout << "Maximum number of MFVB iterations reached." << "\n";
    }

  }

  return Rcpp::List::create(Rcpp::Named("q_beta_params")=Rcpp::List::create(Rcpp::Named("mu_q_beta")=x1,
                                                                            Rcpp::Named("Sigma_q_beta")=Asup11),
                            Rcpp::Named("q_uL1_params")=Rcpp::List::create(Rcpp::Named("mu_q_uL1")=x2i,
                                                                           Rcpp::Named("Sigma_q_uL1")=Asup22i),
                            Rcpp::Named("q_uL2_params")=Rcpp::List::create(Rcpp::Named("mu_q_uL2")=x2ij,
                                                                           Rcpp::Named("Sigma_q_uL2")=Asup22ij),
                            Rcpp::Named("q_sigsq_params")=Rcpp::List::create(Rcpp::Named("xi_q_sigsq")=xi_q_sigsq,
                                                                             Rcpp::Named("lambda_q_sigsq")=lambda_q_sigsq),
                            Rcpp::Named("q_tausq_params")=Rcpp::List::create(Rcpp::Named("xi_q_tausq")=xi_q_tausq,
                                                                             Rcpp::Named("lambda_q_tausq")=lambda_q_tausq),
                            Rcpp::Named("q_zeta_params")=Rcpp::List::create(Rcpp::Named("mu_q_zeta")=mu_q_zeta,
                                                                            Rcpp::Named("lambda_q_zeta")=(2.0 * mu_q_a_zeta)),
                            Rcpp::Named("q_SigmaL1_params")=Rcpp::List::create(Rcpp::Named("xi_q_SigmaL1")=xi_q_SigmaL1,
                                                                               Rcpp::Named("Lambda_q_SigmaL1")=Lambda_q_SigmaL1),
                            Rcpp::Named("q_SigmaL2_params")=Rcpp::List::create(Rcpp::Named("xi_q_SigmaL2")=xi_q_SigmaL2,
                                                                               Rcpp::Named("Lambda_q_SigmaL2")=Lambda_q_SigmaL2));
}
