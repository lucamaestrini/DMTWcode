# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

Lentz_Q <- function(x) {
    .Call('_DMTWpackage_Lentz_Q', PACKAGE = 'DMTWpackage', x)
}

Lentz_R_nu <- function(x, nu) {
    .Call('_DMTWpackage_Lentz_R_nu', PACKAGE = 'DMTWpackage', x, nu)
}

NaiveMfvbLmmThreeLevelGaussPrior <- function(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, maxIter, verbose) {
    .Call('_DMTWpackage_NaiveMfvbLmmThreeLevelGaussPrior', PACKAGE = 'DMTWpackage', yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, maxIter, verbose)
}

NaiveMfvbLmmThreeLevelGlobLocPrior <- function(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, FactorizeZeta, PriorType, maxIter, verbose) {
    .Call('_DMTWpackage_NaiveMfvbLmmThreeLevelGlobLocPrior', PACKAGE = 'DMTWpackage', yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, oMat, nVec, FactorizeZeta, PriorType, maxIter, verbose)
}

NaiveMfvbLmmTwoLevelGaussPrior <- function(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, oVec, maxIter, verbose) {
    .Call('_DMTWpackage_NaiveMfvbLmmTwoLevelGaussPrior', PACKAGE = 'DMTWpackage', yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, oVec, maxIter, verbose)
}

NaiveMfvbLmmTwoLevelGlobLocPrior <- function(yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, oVec, FactorizeZeta, PriorType, maxIter, verbose) {
    .Call('_DMTWpackage_NaiveMfvbLmmTwoLevelGlobLocPrior', PACKAGE = 'DMTWpackage', yvec, XRmat, XAmat, XSmat, Zmat, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, oVec, FactorizeZeta, PriorType, maxIter, verbose)
}

SolveThreeLevelSparseMatrix <- function(a1, Ainf11, a2i, Ainf22i, Ainf12i, a2ij, Ainf22ij, Ainf12ij, Ainf12iandj, x1, Asup11, x2i, Asup22i, Asup12i, x2ij, Asup22ij, Asup12ij, Asup12iandj, otildeMat) {
    invisible(.Call('_DMTWpackage_SolveThreeLevelSparseMatrix', PACKAGE = 'DMTWpackage', a1, Ainf11, a2i, Ainf22i, Ainf12i, a2ij, Ainf22ij, Ainf12ij, Ainf12iandj, x1, Asup11, x2i, Asup22i, Asup12i, x2ij, Asup22ij, Asup12ij, Asup12iandj, otildeMat))
}

SolveTwoLevelSparseMatrix <- function(a1, Ainf11, a2i, Ainf22i, Ainf12i, x1, Asup11, x2i, Asup22i, Asup12i, otildeVec) {
    invisible(.Call('_DMTWpackage_SolveTwoLevelSparseMatrix', PACKAGE = 'DMTWpackage', a1, Ainf11, a2i, Ainf22i, Ainf12i, x1, Asup11, x2i, Asup22i, Asup12i, otildeVec))
}

StreamlinedMfvbLmmThreeLevelGaussPrior <- function(yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, maxIter, verbose) {
    .Call('_DMTWpackage_StreamlinedMfvbLmmThreeLevelGaussPrior', PACKAGE = 'DMTWpackage', yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, maxIter, verbose)
}

StreamlinedMfvbLmmThreeLevelGlobLocPrior <- function(yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose) {
    .Call('_DMTWpackage_StreamlinedMfvbLmmThreeLevelGlobLocPrior', PACKAGE = 'DMTWpackage', yvecField, XmatField, ZL1matField, ZL2matField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigmaL1, nuSigmaL2, sSigmaL1, sSigmaL2, nVec, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose)
}

StreamlinedMfvbLmmTwoLevelGaussPrior <- function(yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, pR, pA, pS, maxIter, verbose) {
    .Call('_DMTWpackage_StreamlinedMfvbLmmTwoLevelGaussPrior', PACKAGE = 'DMTWpackage', yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, nuSigma, sSigma, pR, pA, pS, maxIter, verbose)
}

StreamlinedMfvbLmmTwoLevelGlobLocPrior <- function(yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose) {
    .Call('_DMTWpackage_StreamlinedMfvbLmmTwoLevelGlobLocPrior', PACKAGE = 'DMTWpackage', yvecField, XmatField, ZmatField, muBetaRandA, SigmaBetaRandA, nusigma2, ssigma2, stau2, lambda, nuSigma, sSigma, pR, pA, pS, FactorizeZeta, PriorType, maxIter, verbose)
}

