#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// R_nu function defined in (A.4) of Neville, Ormerod and Wand (2014)
// and implemented with Algorithm 4 of the same paper

// [[Rcpp::export]]
double Lentz_R_nu(const double x, const double nu) {
  // x > 0, nu > 0

  const double eps1 = 1e-30;
  const double eps2 = 1e-7;

  double fprev = eps1;    double Cprev = eps1;
  double Dprev = 0.0;     double Delta = 2.0 + eps2;
  double j = 1.0;

  double Dcurr, Ccurr, fcurr;

  while(fabs(Delta - 1.0) >= eps2){
    j++;

    Dcurr = x + (nu + j) * Dprev;
    Ccurr = x + (nu + j) / Cprev;
    Dcurr = 1.0/Dcurr;
    Delta = Ccurr * Dcurr;
    fcurr = fprev * Delta;

    fprev = fcurr;      Cprev = Ccurr;      Dprev = Dcurr;
  }

  return(1.0/(x + fcurr));
}
