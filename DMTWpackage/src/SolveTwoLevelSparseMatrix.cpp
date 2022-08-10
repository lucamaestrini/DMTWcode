#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void SolveTwoLevelSparseMatrix(arma::vec a1,
                               arma::mat Ainf11,
                               arma::field<arma::vec> a2i,
                               arma::field<arma::mat> Ainf22i,
                               arma::field<arma::mat> Ainf12i,
                               arma::vec& x1,
                               arma::mat& Asup11,
                               arma::field<arma::vec>& x2i,
                               arma::field<arma::mat>& Asup22i,
                               arma::field<arma::mat>& Asup12i,
                               const arma::mat otildeVec)
{

  const int m = otildeVec.n_elem;

  arma::vec omega42 = a1;
  arma::mat Omega43 = Ainf11;

  for(int i = 0; i < m; i++){
    omega42 = omega42 - Ainf12i(i) * solve(Ainf22i(i), a2i(i));
    Omega43 = Omega43 - Ainf12i(i) * solve(Ainf22i(i), trans(Ainf12i(i)));
  }

  Asup11 = inv_sympd(Omega43);
  x1 = Asup11 * omega42;

  for(int i = 0; i < m; i++){
    x2i(i) = solve(Ainf22i(i), (a2i(i) - trans(Ainf12i(i)) * x1));
    Asup12i(i) = -trans(solve(Ainf22i(i), trans(Ainf12i(i))) * Asup11);
    Asup22i(i) = inv_sympd(Ainf22i(i)) - solve(Ainf22i(i), trans(Ainf12i(i)) * Asup12i(i));
  }
}
