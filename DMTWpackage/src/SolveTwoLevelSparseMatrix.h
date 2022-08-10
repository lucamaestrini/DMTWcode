#ifndef SOLVETWOLEVELSPARSEMATRIX
   #define SOLVETWOLEVELSPARSEMATRIX
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
                                  const arma::mat otildeVec);
#endif
