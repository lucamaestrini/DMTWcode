// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Lentz_Q
double Lentz_Q(const double x);
RcppExport SEXP _DMTWpackage_Lentz_Q(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(Lentz_Q(x));
    return rcpp_result_gen;
END_RCPP
}
// Lentz_R_nu
double Lentz_R_nu(const double x, const double nu);
RcppExport SEXP _DMTWpackage_Lentz_R_nu(SEXP xSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(Lentz_R_nu(x, nu));
    return rcpp_result_gen;
END_RCPP
}
// NaiveMfvbLmmThreeLevelGaussPrior
Rcpp::List NaiveMfvbLmmThreeLevelGaussPrior(const arma::vec yvec, const arma::mat XRmat, const arma::mat XAmat, const arma::mat XSmat, const arma::mat Zmat, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double nuSigmaL1, const double nuSigmaL2, const arma::vec sSigmaL1, const arma::vec sSigmaL2, const arma::mat oMat, const arma::vec nVec, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_NaiveMfvbLmmThreeLevelGaussPrior(SEXP yvecSEXP, SEXP XRmatSEXP, SEXP XAmatSEXP, SEXP XSmatSEXP, SEXP ZmatSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP nuSigmaL1SEXP, SEXP nuSigmaL2SEXP, SEXP sSigmaL1SEXP, SEXP sSigmaL2SEXP, SEXP oMatSEXP, SEXP nVecSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XRmat(XRmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XAmat(XAmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XSmat(XSmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Zmat(ZmatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL1(nuSigmaL1SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL2(nuSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL1(sSigmaL1SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL2(sSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type oMat(oMatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type nVec(nVecSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(NaiveMfvbLmmThreeLevelGaussPrior(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// NaiveMfvbLmmThreeLevelGlobLocPrior
Rcpp::List NaiveMfvbLmmThreeLevelGlobLocPrior(const arma::vec yvec, const arma::mat XRmat, const arma::mat XAmat, const arma::mat XSmat, const arma::mat Zmat, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double stau2, const double lambda, const double nuSigmaL1, const double nuSigmaL2, const arma::vec sSigmaL1, const arma::vec sSigmaL2, const arma::mat oMat, const arma::vec nVec, bool FactorizeZeta, const std::string PriorType, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_NaiveMfvbLmmThreeLevelGlobLocPrior(SEXP yvecSEXP, SEXP XRmatSEXP, SEXP XAmatSEXP, SEXP XSmatSEXP, SEXP ZmatSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP stau2SEXP, SEXP lambdaSEXP, SEXP nuSigmaL1SEXP, SEXP nuSigmaL2SEXP, SEXP sSigmaL1SEXP, SEXP sSigmaL2SEXP, SEXP oMatSEXP, SEXP nVecSEXP, SEXP FactorizeZetaSEXP, SEXP PriorTypeSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XRmat(XRmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XAmat(XAmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XSmat(XSmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Zmat(ZmatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type stau2(stau2SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL1(nuSigmaL1SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL2(nuSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL1(sSigmaL1SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL2(sSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type oMat(oMatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type nVec(nVecSEXP);
    Rcpp::traits::input_parameter< bool >::type FactorizeZeta(FactorizeZetaSEXP);
    Rcpp::traits::input_parameter< const std::string >::type PriorType(PriorTypeSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(NaiveMfvbLmmThreeLevelGlobLocPrior(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, FactorizeZeta, PriorType, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// NaiveMfvbLmmTwoLevelGaussPrior
Rcpp::List NaiveMfvbLmmTwoLevelGaussPrior(const arma::vec yvec, const arma::mat XRmat, const arma::mat XAmat, const arma::mat XSmat, const arma::mat Zmat, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double nuSigma, const arma::vec sSigma, const arma::vec oVec, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_NaiveMfvbLmmTwoLevelGaussPrior(SEXP yvecSEXP, SEXP XRmatSEXP, SEXP XAmatSEXP, SEXP XSmatSEXP, SEXP ZmatSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP nuSigmaSEXP, SEXP sSigmaSEXP, SEXP oVecSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XRmat(XRmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XAmat(XAmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XSmat(XSmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Zmat(ZmatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigma(nuSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigma(sSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type oVec(oVecSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(NaiveMfvbLmmTwoLevelGaussPrior(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, oVec, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// NaiveMfvbLmmTwoLevelGlobLocPrior
Rcpp::List NaiveMfvbLmmTwoLevelGlobLocPrior(const arma::vec yvec, const arma::mat XRmat, const arma::mat XAmat, const arma::mat XSmat, const arma::mat Zmat, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double stau2, const double lambda, const double nuSigma, const arma::vec sSigma, const arma::vec oVec, bool FactorizeZeta, const std::string PriorType, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_NaiveMfvbLmmTwoLevelGlobLocPrior(SEXP yvecSEXP, SEXP XRmatSEXP, SEXP XAmatSEXP, SEXP XSmatSEXP, SEXP ZmatSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP stau2SEXP, SEXP lambdaSEXP, SEXP nuSigmaSEXP, SEXP sSigmaSEXP, SEXP oVecSEXP, SEXP FactorizeZetaSEXP, SEXP PriorTypeSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XRmat(XRmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XAmat(XAmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type XSmat(XSmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Zmat(ZmatSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type stau2(stau2SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigma(nuSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigma(sSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type oVec(oVecSEXP);
    Rcpp::traits::input_parameter< bool >::type FactorizeZeta(FactorizeZetaSEXP);
    Rcpp::traits::input_parameter< const std::string >::type PriorType(PriorTypeSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(NaiveMfvbLmmTwoLevelGlobLocPrior(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, oVec, FactorizeZeta, PriorType, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// SolveThreeLevelSparseMatrix
void SolveThreeLevelSparseMatrix(arma::vec a1, arma::mat Ainf11, arma::field<arma::vec> a2i, arma::field<arma::mat> Ainf22i, arma::field<arma::mat> Ainf12i, arma::field<arma::vec> a2ij, arma::field<arma::mat> Ainf22ij, arma::field<arma::mat> Ainf12ij, arma::field<arma::mat> Ainf12iandj, arma::vec& x1, arma::mat& Asup11, arma::field<arma::vec>& x2i, arma::field<arma::mat>& Asup22i, arma::field<arma::mat>& Asup12i, arma::field<arma::vec>& x2ij, arma::field<arma::mat>& Asup22ij, arma::field<arma::mat>& Asup12ij, arma::field<arma::mat>& Asup12iandj, const arma::mat otildeMat);
RcppExport SEXP _DMTWpackage_SolveThreeLevelSparseMatrix(SEXP a1SEXP, SEXP Ainf11SEXP, SEXP a2iSEXP, SEXP Ainf22iSEXP, SEXP Ainf12iSEXP, SEXP a2ijSEXP, SEXP Ainf22ijSEXP, SEXP Ainf12ijSEXP, SEXP Ainf12iandjSEXP, SEXP x1SEXP, SEXP Asup11SEXP, SEXP x2iSEXP, SEXP Asup22iSEXP, SEXP Asup12iSEXP, SEXP x2ijSEXP, SEXP Asup22ijSEXP, SEXP Asup12ijSEXP, SEXP Asup12iandjSEXP, SEXP otildeMatSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type a1(a1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Ainf11(Ainf11SEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type a2i(a2iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf22i(Ainf22iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf12i(Ainf12iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type a2ij(a2ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf22ij(Ainf22ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf12ij(Ainf12ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf12iandj(Ainf12iandjSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Asup11(Asup11SEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type x2i(x2iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup22i(Asup22iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup12i(Asup12iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type x2ij(x2ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup22ij(Asup22ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup12ij(Asup12ijSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup12iandj(Asup12iandjSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type otildeMat(otildeMatSEXP);
    SolveThreeLevelSparseMatrix(a1, Ainf11, a2i, Ainf22i, Ainf12i, a2ij, Ainf22ij, Ainf12ij, Ainf12iandj, x1, Asup11, x2i, Asup22i, Asup12i, x2ij, Asup22ij, Asup12ij, Asup12iandj, otildeMat);
    return R_NilValue;
END_RCPP
}
// SolveTwoLevelSparseMatrix
void SolveTwoLevelSparseMatrix(arma::vec a1, arma::mat Ainf11, arma::field<arma::vec> a2i, arma::field<arma::mat> Ainf22i, arma::field<arma::mat> Ainf12i, arma::vec& x1, arma::mat& Asup11, arma::field<arma::vec>& x2i, arma::field<arma::mat>& Asup22i, arma::field<arma::mat>& Asup12i, const arma::mat otildeVec);
RcppExport SEXP _DMTWpackage_SolveTwoLevelSparseMatrix(SEXP a1SEXP, SEXP Ainf11SEXP, SEXP a2iSEXP, SEXP Ainf22iSEXP, SEXP Ainf12iSEXP, SEXP x1SEXP, SEXP Asup11SEXP, SEXP x2iSEXP, SEXP Asup22iSEXP, SEXP Asup12iSEXP, SEXP otildeVecSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type a1(a1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Ainf11(Ainf11SEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type a2i(a2iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf22i(Ainf22iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Ainf12i(Ainf12iSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Asup11(Asup11SEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type x2i(x2iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup22i(Asup22iSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type Asup12i(Asup12iSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type otildeVec(otildeVecSEXP);
    SolveTwoLevelSparseMatrix(a1, Ainf11, a2i, Ainf22i, Ainf12i, x1, Asup11, x2i, Asup22i, Asup12i, otildeVec);
    return R_NilValue;
END_RCPP
}
// StreamlinedMfvbLmmThreeLevelGaussPrior
Rcpp::List StreamlinedMfvbLmmThreeLevelGaussPrior(const arma::field<arma::vec> yvecField, const arma::field<arma::mat> XmatField, const arma::field<arma::mat> ZL1matField, const arma::field<arma::mat> ZL2matField, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double nuSigmaL1, const double nuSigmaL2, const arma::vec sSigmaL1, const arma::vec sSigmaL2, const arma::vec nVec, const int pR, const int pA, const int pS, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_StreamlinedMfvbLmmThreeLevelGaussPrior(SEXP yvecFieldSEXP, SEXP XmatFieldSEXP, SEXP ZL1matFieldSEXP, SEXP ZL2matFieldSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP nuSigmaL1SEXP, SEXP nuSigmaL2SEXP, SEXP sSigmaL1SEXP, SEXP sSigmaL2SEXP, SEXP nVecSEXP, SEXP pRSEXP, SEXP pASEXP, SEXP pSSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec> >::type yvecField(yvecFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type XmatField(XmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZL1matField(ZL1matFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZL2matField(ZL2matFieldSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL1(nuSigmaL1SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL2(nuSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL1(sSigmaL1SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL2(sSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type nVec(nVecSEXP);
    Rcpp::traits::input_parameter< const int >::type pR(pRSEXP);
    Rcpp::traits::input_parameter< const int >::type pA(pASEXP);
    Rcpp::traits::input_parameter< const int >::type pS(pSSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(StreamlinedMfvbLmmThreeLevelGaussPrior(yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// StreamlinedMfvbLmmThreeLevelGlobLocPrior
Rcpp::List StreamlinedMfvbLmmThreeLevelGlobLocPrior(const arma::field<arma::vec> yvecField, const arma::field<arma::mat> XmatField, const arma::field<arma::mat> ZL1matField, const arma::field<arma::mat> ZL2matField, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double stau2, const double lambda, const double nuSigmaL1, const double nuSigmaL2, const arma::vec sSigmaL1, const arma::vec sSigmaL2, const arma::vec nVec, const int pR, const int pA, const int pS, bool FactorizeZeta, const std::string PriorType, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_StreamlinedMfvbLmmThreeLevelGlobLocPrior(SEXP yvecFieldSEXP, SEXP XmatFieldSEXP, SEXP ZL1matFieldSEXP, SEXP ZL2matFieldSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP stau2SEXP, SEXP lambdaSEXP, SEXP nuSigmaL1SEXP, SEXP nuSigmaL2SEXP, SEXP sSigmaL1SEXP, SEXP sSigmaL2SEXP, SEXP nVecSEXP, SEXP pRSEXP, SEXP pASEXP, SEXP pSSEXP, SEXP FactorizeZetaSEXP, SEXP PriorTypeSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec> >::type yvecField(yvecFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type XmatField(XmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZL1matField(ZL1matFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZL2matField(ZL2matFieldSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type stau2(stau2SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL1(nuSigmaL1SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigmaL2(nuSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL1(sSigmaL1SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigmaL2(sSigmaL2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type nVec(nVecSEXP);
    Rcpp::traits::input_parameter< const int >::type pR(pRSEXP);
    Rcpp::traits::input_parameter< const int >::type pA(pASEXP);
    Rcpp::traits::input_parameter< const int >::type pS(pSSEXP);
    Rcpp::traits::input_parameter< bool >::type FactorizeZeta(FactorizeZetaSEXP);
    Rcpp::traits::input_parameter< const std::string >::type PriorType(PriorTypeSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(StreamlinedMfvbLmmThreeLevelGlobLocPrior(yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// StreamlinedMfvbLmmTwoLevelGaussPrior
Rcpp::List StreamlinedMfvbLmmTwoLevelGaussPrior(const arma::field<arma::vec> yvecField, const arma::field<arma::mat> XmatField, const arma::field<arma::mat> ZmatField, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double nuSigma, const arma::vec sSigma, const int pR, const int pA, const int pS, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_StreamlinedMfvbLmmTwoLevelGaussPrior(SEXP yvecFieldSEXP, SEXP XmatFieldSEXP, SEXP ZmatFieldSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP nuSigmaSEXP, SEXP sSigmaSEXP, SEXP pRSEXP, SEXP pASEXP, SEXP pSSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec> >::type yvecField(yvecFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type XmatField(XmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZmatField(ZmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigma(nuSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigma(sSigmaSEXP);
    Rcpp::traits::input_parameter< const int >::type pR(pRSEXP);
    Rcpp::traits::input_parameter< const int >::type pA(pASEXP);
    Rcpp::traits::input_parameter< const int >::type pS(pSSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(StreamlinedMfvbLmmTwoLevelGaussPrior(yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, pR, pA, pS, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// StreamlinedMfvbLmmTwoLevelGlobLocPrior
Rcpp::List StreamlinedMfvbLmmTwoLevelGlobLocPrior(const arma::field<arma::vec> yvecField, const arma::field<arma::mat> XmatField, const arma::field<arma::mat> ZmatField, const arma::vec muBetaRandA, const arma::mat SigmaBetaRandA, const double nusigma2, const double ssigma2, const double stau2, const double lambda, const double nuSigma, const arma::vec sSigma, const int pR, const int pA, const int pS, bool FactorizeZeta, const std::string PriorType, const int maxIter, bool verbose);
RcppExport SEXP _DMTWpackage_StreamlinedMfvbLmmTwoLevelGlobLocPrior(SEXP yvecFieldSEXP, SEXP XmatFieldSEXP, SEXP ZmatFieldSEXP, SEXP muBetaRandASEXP, SEXP SigmaBetaRandASEXP, SEXP nusigma2SEXP, SEXP ssigma2SEXP, SEXP stau2SEXP, SEXP lambdaSEXP, SEXP nuSigmaSEXP, SEXP sSigmaSEXP, SEXP pRSEXP, SEXP pASEXP, SEXP pSSEXP, SEXP FactorizeZetaSEXP, SEXP PriorTypeSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::vec> >::type yvecField(yvecFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type XmatField(XmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat> >::type ZmatField(ZmatFieldSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type muBetaRandA(muBetaRandASEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type SigmaBetaRandA(SigmaBetaRandASEXP);
    Rcpp::traits::input_parameter< const double >::type nusigma2(nusigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type ssigma2(ssigma2SEXP);
    Rcpp::traits::input_parameter< const double >::type stau2(stau2SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type nuSigma(nuSigmaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type sSigma(sSigmaSEXP);
    Rcpp::traits::input_parameter< const int >::type pR(pRSEXP);
    Rcpp::traits::input_parameter< const int >::type pA(pASEXP);
    Rcpp::traits::input_parameter< const int >::type pS(pSSEXP);
    Rcpp::traits::input_parameter< bool >::type FactorizeZeta(FactorizeZetaSEXP);
    Rcpp::traits::input_parameter< const std::string >::type PriorType(PriorTypeSEXP);
    Rcpp::traits::input_parameter< const int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(StreamlinedMfvbLmmTwoLevelGlobLocPrior(yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DMTWpackage_Lentz_Q", (DL_FUNC) &_DMTWpackage_Lentz_Q, 1},
    {"_DMTWpackage_Lentz_R_nu", (DL_FUNC) &_DMTWpackage_Lentz_R_nu, 2},
    {"_DMTWpackage_NaiveMfvbLmmThreeLevelGaussPrior", (DL_FUNC) &_DMTWpackage_NaiveMfvbLmmThreeLevelGaussPrior, 17},
    {"_DMTWpackage_NaiveMfvbLmmThreeLevelGlobLocPrior", (DL_FUNC) &_DMTWpackage_NaiveMfvbLmmThreeLevelGlobLocPrior, 21},
    {"_DMTWpackage_NaiveMfvbLmmTwoLevelGaussPrior", (DL_FUNC) &_DMTWpackage_NaiveMfvbLmmTwoLevelGaussPrior, 14},
    {"_DMTWpackage_NaiveMfvbLmmTwoLevelGlobLocPrior", (DL_FUNC) &_DMTWpackage_NaiveMfvbLmmTwoLevelGlobLocPrior, 18},
    {"_DMTWpackage_SolveThreeLevelSparseMatrix", (DL_FUNC) &_DMTWpackage_SolveThreeLevelSparseMatrix, 19},
    {"_DMTWpackage_SolveTwoLevelSparseMatrix", (DL_FUNC) &_DMTWpackage_SolveTwoLevelSparseMatrix, 11},
    {"_DMTWpackage_StreamlinedMfvbLmmThreeLevelGaussPrior", (DL_FUNC) &_DMTWpackage_StreamlinedMfvbLmmThreeLevelGaussPrior, 18},
    {"_DMTWpackage_StreamlinedMfvbLmmThreeLevelGlobLocPrior", (DL_FUNC) &_DMTWpackage_StreamlinedMfvbLmmThreeLevelGlobLocPrior, 22},
    {"_DMTWpackage_StreamlinedMfvbLmmTwoLevelGaussPrior", (DL_FUNC) &_DMTWpackage_StreamlinedMfvbLmmTwoLevelGaussPrior, 14},
    {"_DMTWpackage_StreamlinedMfvbLmmTwoLevelGlobLocPrior", (DL_FUNC) &_DMTWpackage_StreamlinedMfvbLmmTwoLevelGlobLocPrior, 18},
    {NULL, NULL, 0}
};

RcppExport void R_init_DMTWpackage(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
