#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Q function defined in (A.3) of Neville, Ormerod and Wand (2014)
// and implemented with Algorithm 2 of the same paper

// [[Rcpp::export]]
double Lentz_Q(const double x) {
  // x > 0

  const double eps1 = 1e-30;
  const double eps2 = 1e-7;

  double fprev = eps1;    double Cprev = eps1;
  double Dprev = 0.0;     double Delta = 2.0 + eps2;
  double j = 1.0;

  double Dcurr, Ccurr, fcurr;

  while(fabs(Delta - 1.0) >= eps2){
    j++;

    Dcurr = x + (2.0 * j) - 1 - pow(j - 1.0, 2.0) * Dprev;
    Ccurr = x + (2.0 * j) - 1 - pow(j - 1.0, 2.0) / Cprev;
    Dcurr = 1.0/Dcurr;
    Delta = Ccurr * Dcurr;
    fcurr = fprev * Delta;

    fprev = fcurr;      Cprev = Ccurr;      Dprev = Dcurr;
  }

  return(1.0/(x + 1 + fcurr));
}
