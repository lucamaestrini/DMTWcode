#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void SolveThreeLevelSparseMatrix(arma::vec a1,
                                 arma::mat Ainf11,
                                 arma::field<arma::vec> a2i,
                                 arma::field<arma::mat> Ainf22i,
                                 arma::field<arma::mat> Ainf12i,
                                 arma::field<arma::vec> a2ij,
                                 arma::field<arma::mat> Ainf22ij,
                                 arma::field<arma::mat> Ainf12ij,
                                 arma::field<arma::mat> Ainf12iandj,
                                 arma::vec& x1,
                                 arma::mat& Asup11,
                                 arma::field<arma::vec>& x2i,
                                 arma::field<arma::mat>& Asup22i,
                                 arma::field<arma::mat>& Asup12i,
                                 arma::field<arma::vec>& x2ij,
                                 arma::field<arma::mat>& Asup22ij,
                                 arma::field<arma::mat>& Asup12ij,
                                 arma::field<arma::mat>& Asup12iandj,
                                 const arma::mat otildeMat)
{

  const int m = otildeMat.n_rows;
  arma::vec n(m); for(int i = 0; i < m; i++) n(i) = sum(otildeMat.row(i) > 0);

  arma::field<arma::vec> h2i(m);
  arma::field<arma::mat> H12i(m);
  arma::field<arma::mat> H22i(m);

  arma::vec omega46 = a1;
  arma::mat Omega47 = Ainf11;

  for(int i = 0; i < m; i++){
    h2i(i) = a2i(i);
    H12i(i) = Ainf12i(i);
    H22i(i) = Ainf22i(i);
    for(int j = 0; j < n(i); j++){
      h2i(i) = h2i(i) - Ainf12iandj(i,j) * solve(Ainf22ij(i,j), a2ij(i,j));
      H12i(i) = H12i(i) - Ainf12ij(i,j) * solve(Ainf22ij(i,j), trans(Ainf12iandj(i,j)));
      H22i(i) = H22i(i) - Ainf12iandj(i,j) * solve(Ainf22ij(i,j), trans(Ainf12iandj(i,j)));
      omega46 = omega46 - Ainf12ij(i,j) * solve(Ainf22ij(i,j), a2ij(i,j));
      Omega47 = Omega47 - Ainf12ij(i,j) * solve(Ainf22ij(i,j), trans(Ainf12ij(i,j)));
    }
    omega46 = omega46 - H12i(i) * solve(H22i(i), h2i(i));
    Omega47 = Omega47 - H12i(i) * solve(H22i(i), trans(H12i(i)));
  }

  Asup11 = inv_sympd(Omega47);
  x1 = Asup11 * omega46;

  for(int i = 0; i < m; i++){
    x2i(i) = solve(H22i(i), (h2i(i) - trans(H12i(i)) * x1));
    Asup12i(i) = -trans(solve(H22i(i), trans(H12i(i))) * Asup11);
    Asup22i(i) = inv_sympd(H22i(i)) - solve(H22i(i), trans(H12i(i)) * Asup12i(i));
    for(int j = 0; j < n(i); j++){
      x2ij(i,j) = solve(Ainf22ij(i,j), (a2ij(i,j) - trans(Ainf12ij(i,j)) * x1 - trans(Ainf12iandj(i,j)) * x2i(i)));
      Asup12ij(i,j) = -trans(solve(Ainf22ij(i,j), trans(Ainf12ij(i,j)) * Asup11 + trans(Ainf12iandj(i,j)) * trans(Asup12i(i))));
      Asup12iandj(i,j) = -trans(solve(Ainf22ij(i,j), trans(Ainf12ij(i,j)) * Asup12i(i) + trans(Ainf12iandj(i,j)) * Asup22i(i)));
      Asup22ij(i,j) = inv_sympd(Ainf22ij(i,j)) - solve(Ainf22ij(i,j), trans(Ainf12ij(i,j)) * Asup12ij(i,j) + trans(Ainf12iandj(i,j)) * Asup12iandj(i,j));
    }
  }
}
