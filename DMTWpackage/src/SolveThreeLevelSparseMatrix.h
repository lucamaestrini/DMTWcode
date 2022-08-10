#ifndef SOLVETHREELEVELSPARSEMATRIX
   #define SOLVETHREELEVELSPARSEMATRIX
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
                                    const arma::mat otildeMat);
#endif
