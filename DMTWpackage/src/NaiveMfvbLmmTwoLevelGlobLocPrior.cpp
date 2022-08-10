#include <RcppArmadillo.h>
#include "Lentz_Q.h"
#include "Lentz_R_nu.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List NaiveMfvbLmmTwoLevelGlobLocPrior(const arma::vec yvec,
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
                                            const double nuSigma,
                                            const arma::vec sSigma,
                                            const arma::vec oVec,
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
  // nuSigma: degrees of freedom hyperparameters for Sigma
  // sSigma: vector of scale hyperparameters for Sigma
  // oVec: vector of dimension m, containing all the o_i's for all 1 \le i \le m
  // FactorizeZeta: boolean, whether for considering q(zeta_h) q(a_{zeta_h}) (TRUE)
  //                or q(zeta_h) (FALSE) without introducing the augmented a_{zeta_h}
  // PriorType: string for the type of global local specification. one between
  //            'Laplace', 'Horseshoe' or 'Normal-Exponential-Gamma'.
  // maxIter: number of iterations over which to run the algorithm
  // verbose: boolean, whether for printing the evolution of iterations

  // extract model dimensions:

  const int n = yvec.n_elem;
  const int m = oVec.n_elem;
  const int pR = XRmat.n_cols;
  const int pA = XAmat.n_cols;
  const int pS = XSmat.n_cols;
  const int q = sSigma.n_elem;
  const int ncZ = m * q;
  const int p = pR + pA + pS;

  // building useful matrices:

  const arma::mat Xmat = join_horiz(XRmat, XAmat, XSmat);
  const arma::mat Cmat = join_horiz(Xmat, Zmat);
  arma::mat CTC(p + ncZ, p + ncZ, fill::zeros);
  arma::vec CTy(n, fill::zeros);
  arma::mat DMFVB(p + ncZ, p + ncZ, fill::zeros);
  arma::vec oMFVB(p + ncZ, fill::zeros);

  // build useful vectors for row- and column-indexing:

  arma::vec indcolZmat(m + 1, fill::zeros);
  indcolZmat.tail(m) = regspace<vec>(q, q, q * m);
  arma::vec indrowZmat(m + 1, fill::zeros);
  indrowZmat.tail(m) = oVec;
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
  arma::mat Lambda_q_Sigma(q, q, fill::zeros);
  arma::mat Lambda_q_ASigma(q, q, fill::zeros);
  arma::mat M_q_recip_Sigma(q, q, fill::eye);
  arma::mat M_q_recip_ASigma(q, q, fill::eye);
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
  arma::mat sumMatrix(q, q, fill::zeros);
  arma::vec mu_q_u_toSum(q, fill::zeros);
  arma::mat Sigma_q_u_toSum(q, q, fill::zeros);

  ///////////////////
  // Fixed updates //
  ///////////////////

  const double xi_q_sigsq = n + nusigma2;
  const double xi_q_a_sigsq = nusigma2 + 1.0;
  const double xi_q_tausq = pS + 1.0;
  const double xi_q_a_tausq = 2.0;
  const double xi_q_Sigma = nuSigma + m + 2.0 * q - 2.0;
  const double xi_q_ASigma = nuSigma + q;

  bool converged = false;
  int itNum = 0;

  while(!converged){

    itNum++;

    /////////////////////////////
    // Updates for q*(beta, u) //
    // being a N(mu, Sigma)    //
    /////////////////////////////

    DMFVB.submat(p-pS, p-pS, p-1, p-1).diag() = mu_q_recip_tausq * mu_q_zeta;
    DMFVB.submat(span(p, p + ncZ-1), span(p, p + ncZ-1)) = kron(eye<mat>(m, m), M_q_recip_Sigma);

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
    // Gamma if HorseShoe                                       //
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
    // Updates for q*(Sigma)                          //
    // being an Inverse-G-Wishart(G_full, xi, Lambda) //
    ////////////////////////////////////////////////////

    Lambda_q_Sigma = M_q_recip_ASigma;
    for(int i = 0; i < m; i++){
      Lambda_q_Sigma = Lambda_q_Sigma + mu_q_ui(i).subvec(0, q-1) * trans(mu_q_ui(i).subvec(0, q-1)) + Sigma_q_ui(i).submat(0, 0, q-1, q-1);
    }
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
                            Rcpp::Named("q_Sigma_params")=Rcpp::List::create(Rcpp::Named("xi_q_Sigma")=xi_q_Sigma,
                                                                             Rcpp::Named("Lambda_q_Sigma")=Lambda_q_Sigma));
}
