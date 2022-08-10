#include <RcppArmadillo.h>
#include "Lentz_Q.h"
#include "Lentz_R_nu.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List NaiveMfvbLmmThreeLevelGlobLocPrior(const arma::vec yvec,
                                              const arma::mat XRmat,
                                              const arma::mat XAmat,
                                              const arma::mat XSmat,
                                              const arma::mat Zmat,
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
                                              const arma::mat oMat,
                                              const arma::vec nVec,
                                              bool FactorizeZeta,
                                              const std::string PriorType,
                                              const int maxIter,
                                              bool verbose) {

  // yvec: response variable vector
  // XRmat: design matrix for fixed effects associated to random effects
  // XAmat: design matrix for additional fixed effects
  // XSmat: design matrix for fixed effects subject to global-local prior
  // Zmat: design matrix for random effects
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
  // oMat: matrix of dimension m x max(n_i), containing all the o_{ij}s
  //       for all 1 \le j \le n_i, 1 \le i \le m
  // nVec: vector of dimension m contanining all the n_i's, for all 1 \le i \le m
  // FactorizeZeta: boolean, whether for considering q(zeta_h) q(a_{zeta_h}) (TRUE)
  //                or q(zeta_h) (FALSE) without introducing the augmented a_{zeta_h}
  // PriorType: string for the type of global local specification. one between
  //            'Laplace', 'Horseshoe' or 'Normal-Exponential-Gamma'.
  // maxIter: number of iterations over which to run the algorithm
  // verbose: boolean, whether for printing the evolution of iterations

  // extract model dimensions:

  const int n = yvec.n_elem;
  const int m = nVec.n_elem;
  const int pR = XRmat.n_cols;
  const int pA = XAmat.n_cols;
  const int pS = XSmat.n_cols;
  const int qL1 = sSigmaL1.n_elem;
  const int qL2 = sSigmaL2.n_elem;
  const int ncZ = sum(qL1 + qL2 * nVec);
  const int p = pR + pA + pS;

  // building useful matrices:

  arma::vec oMatrowsum = sum(oMat, 1);

  const arma::mat Xmat = join_horiz(XRmat, XAmat, XSmat);
  const arma::mat Cmat = join_horiz(Xmat, Zmat);
  arma::mat CTC(p + ncZ, p + ncZ, fill::zeros);
  arma::vec CTy(n, fill::zeros);
  arma::mat DMFVB(p + ncZ, p + ncZ, fill::zeros);
  arma::vec oMFVB(p + ncZ, fill::zeros);

  // build useful vectors for row- and column-indexing:

  arma::vec indcolZmat(m + 1, fill::zeros);
  indcolZmat.tail(m) = qL1 + nVec * qL2;
  indcolZmat = cumsum(indcolZmat);
  arma::vec indrowZmat(m + 1, fill::zeros);
  indrowZmat.tail(m) = oMatrowsum;
  indrowZmat = cumsum(indrowZmat);

  // building CTC matrix exploiting its particular structure:

  CTC(span(0, p-1), span(0, p-1)) = trans(Xmat) * Xmat;
  CTC(span(0, p-1), span(p, p + ncZ-1)) = trans(Xmat) * Zmat;
  CTC(span(p, p + ncZ-1), span(0, p-1)) = trans(Zmat) * Xmat;
  for(int w = 0; w < m; w++){
    CTC(span(p+indcolZmat(w), p+indcolZmat(w+1)-1), span(p+indcolZmat(w), p+indcolZmat(w+1)-1)) =
      trans(Zmat(span(indrowZmat(w), indrowZmat(w+1)-1), span(indcolZmat(w), indcolZmat(w+1)-1))) *
      Zmat(span(indrowZmat(w), indrowZmat(w+1)-1), span(indcolZmat(w), indcolZmat(w+1)-1));
  }

  // building CTy vector:

  CTy = trans(Cmat) * yvec;

  // populate DMFVB and oMFVB:

  DMFVB.submat(0, 0, (p-pS)-1, (p-pS)-1) = inv_sympd(SigmaBetaRandA);
  oMFVB.subvec(0, (p-pS)-1) = solve(SigmaBetaRandA, muBetaRandA);

  // initializing variational parameters:

  arma::vec mu_q_betau(p + ncZ, fill::zeros);
  arma::mat Sigma_q_betau(p + ncZ, p + ncZ, fill::zeros);
  arma::vec mu_q_betaS(pS, fill::zeros);
  arma::mat Sigma_q_betaS(pS, pS, fill::zeros);
  arma::vec mu_q_beta(p, fill::zeros);
  arma::mat Sigma_q_beta(p, p, fill::zeros);
  arma::vec mu_q_u(ncZ, fill::zeros);
  arma::mat Sigma_q_u(ncZ, ncZ, fill::zeros);
  arma::field<arma::vec> mu_q_ui(m);
  arma::field<arma::mat> Sigma_q_ui(m);
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
  arma::mat sumMatrixL1(qL1, qL1, fill::zeros);
  arma::mat sumMatrixL2(qL2, qL2, fill::zeros);
  arma::vec mu_q_uL1_toSum(qL1, fill::zeros);
  arma::mat Sigma_q_uL1_toSum(qL1, qL1, fill::zeros);
  arma::vec mu_q_uL2_toSum(qL2, fill::zeros);
  arma::mat Sigma_q_uL2_toSum(qL2, qL2, fill::zeros);

  ///////////////////
  // Fixed updates //
  ///////////////////

  const double xi_q_sigsq = n + nusigma2;
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

    /////////////////////////////
    // Updates for q*(beta, u) //
    // being a N(mu, Sigma)    //
    /////////////////////////////

    DMFVB.submat(p-pS, p-pS, p-1, p-1).diag() = mu_q_recip_tausq * mu_q_zeta;
    for(int w = 0; w < m; w++){
      arma::mat matToInput(qL1+nVec(w)*qL2, qL1+nVec(w)*qL2, fill::zeros);
      matToInput(span(0, qL1-1), span(0, qL1-1)) = M_q_recip_SigmaL1;
      matToInput(span(qL1, qL1+nVec(w)*qL2-1), span(qL1, qL1+nVec(w)*qL2-1)) = kron(eye<mat>(nVec(w), nVec(w)), M_q_recip_SigmaL2);
      DMFVB.submat(span(p+indcolZmat(w), p+indcolZmat(w+1)-1), span(p+indcolZmat(w), p+indcolZmat(w+1)-1)) = matToInput;
    }

    Sigma_q_betau = inv_sympd(mu_q_recip_sigsq * CTC + DMFVB);
    mu_q_betau = Sigma_q_betau * (mu_q_recip_sigsq * CTy + oMFVB);

    // compute E_q[(beta^S)^2]:

    mu_q_betaS = mu_q_betau(span(p-pS, p-1));
    Sigma_q_betaS = Sigma_q_betau(span(p-pS, p-1), span(p-pS, p-1));

    mu_q_betaSpow2 = pow(mu_q_betaS, 2.0) + Sigma_q_betaS.diag();

    // extract subvectors and submatrices for beta and u, respectively:

    mu_q_beta = mu_q_betau(span(0, p-1));
    Sigma_q_beta = Sigma_q_betau(span(0, p-1), span(0, p-1));
    mu_q_u = mu_q_betau(span(p, (p+ncZ)-1));
    Sigma_q_u = Sigma_q_betau(span(p, (p+ncZ)-1), span(p, (p+ncZ)-1));
    for(int w = 0; w < m; w++){
      mu_q_ui(w) = mu_q_u(span(indcolZmat(w), indcolZmat(w+1)-1));
      Sigma_q_ui(w) = Sigma_q_u(span(indcolZmat(w), indcolZmat(w+1)-1), span(indcolZmat(w), indcolZmat(w+1)-1));
    }

    //////////////////////////////////
    // Updates for q*(sigsq)        //
    // being an Inv-X^2(xi, lambda) //
    /////////////////////////////////

    lambda_q_sigsq = dot(yvec - Cmat * mu_q_betau, yvec - Cmat * mu_q_betau) +
      trace(CTC * Sigma_q_betau) + mu_q_recip_a_sigsq;
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
          mu_q_zeta(h) = 1.0/(G(h) * Lentz_Q(G(h))) - 1;
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

    Lambda_q_SigmaL1 = M_q_recip_ASigmaL1;
    Lambda_q_SigmaL2 = M_q_recip_ASigmaL2;
    for(int i = 0; i < m; i++){
      Lambda_q_SigmaL1 = Lambda_q_SigmaL1 + mu_q_ui(i).subvec(0, qL1-1) * trans(mu_q_ui(i).subvec(0, qL1-1)) + Sigma_q_ui(i).submat(0, 0, qL1-1, qL1-1);
      for(int j = 0; j < nVec(i); j++){
        Lambda_q_SigmaL2 = Lambda_q_SigmaL2 + mu_q_ui(i).subvec(qL1+j*qL2, qL1+j*qL2+1) * trans(mu_q_ui(i).subvec(qL1+j*qL2, qL1+j*qL2+1)) + Sigma_q_ui(i).submat(qL1+j*qL2, qL1+j*qL2, qL1+j*qL2+1, qL1+j*qL2+1);
      }
    }
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

  return Rcpp::List::create(Rcpp::Named("q_beta_params")=Rcpp::List::create(Rcpp::Named("mu_q_beta")=mu_q_beta,
                                                                            Rcpp::Named("Sigma_q_beta")=Sigma_q_beta),
                            Rcpp::Named("q_u_params")=Rcpp::List::create(Rcpp::Named("mu_q_u")=mu_q_u,
                                                                         Rcpp::Named("Sigma_q_u")=Sigma_q_u),
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
