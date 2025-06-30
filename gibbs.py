import Conditionals as cds
import initialize as init
import numpy as np

def RunBinaryGibbs(Y, ns, p, modelType, initType, nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                   thetaTau, phiTau, alphas, randomWalkVariance = 9):
        '''
        Inputs: 
            Y (T x n x n Numpy array of data)
            ns (int number of steps)
            p (int dimension of latent space)
            modelType (either "poisson" or "binary")
            initialization (either "base" or something else that we create later)
            nuIN (mean of prior on betaIN)
            etaIN (variance of prior on betaIN)
            nuOUT (mean of prior on betaOUT)
            etaOUT (variance of prior on betaOUT)
            thetaSigma (shape parameter of prior on SigmaSq)
            phiSigma (scale parameter of prior on SigmaSq)
            thetaTau (shape parameter of prior on TauSq)
            phiTau (scale parameter of prior on TauSq)
            alphas (parameters for Dirichlet prior on r_{1:n})
        
        ** FOR NOW: 
            ** Fix the variance of the normal random walk
            ** Fix the Dirichlet factor (what value we will use in the proposal)
        
        Outputs:
            X (ns x T x n x p Numpy array of latent positions samples from Markov Chain)
            r (ns x n Numpy array of reach samples from Markov Chain)
            tauSq (ns array of tauSq samples from Markov Chain)
            sigmaSq (ns array of sigmaSq samples from Markov Chain)
            betaIN (ns array of betaIN samples from Markov Chain)
            betaOUT (ns array of betaOUT samples from Markov Chain)
        '''

        # Assign the conditionals based on the input argument
        if modelType == "binary":
            conditionals = cds.BinaryConditionals(nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                                                  thetaTau, phiTau, alphas)
        elif modelType == "poisson":
            conditionals = cds.PoissonConditionals(nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                                                   thetaTau, phiTau, alphas)

        # Define key things:
        T = Y.shape[0]
        n = Y.shape[1]

        # Set up empty Numpy arrays
        positions = np.empty(shape=(ns, T, n, p))
        radii = np.empty(shape=(ns, n))
        betaIN = np.empty(shape=(ns))
        betaOUT = np.empty(shape=(ns))
        tauSq = np.empty(shape=(ns))
        sigmaSq = np.empty(shape=(ns))

        # Assign the initialization based on the input argument
        if initType == "base":
            initialization = init.BaseInitialization(Y, positions, radii, betaIN, betaOUT, tauSq, sigmaSq)
        
        # Set initial values for all
        positions, radii, betaIN, betaOUT, tauSq, sigmaSq = initialization.InitializeAll()

        # Setup the currentData dictionary
        currentData = {"Y" : Y, 
                       "X" : positions[0],
                       "r" : radii[0],
                       "betaIN" : betaIN[0],
                       "betaOUT" : betaOUT[0],
                       "tauSq" : tauSq[0],
                       "sigmaSq" : sigmaSq[0]}

        # Define a closure for the SampleFromIntMultivarNormal (to only need to specify the mean for sampling positions)
        def SampleFromIndMultivarNormalFixedVarP(mean):
            return SampleFromIndMultivarNormal(mean, randomWalkVariance, p)
        
        # Define a closure for sampling from the univariate random normal (to fix variance, only need to specify mean) for betaIN, betaOUT
        def SampleFromNormalFixedVar(mean):
            return np.random.normal(mean, np.sqrt(randomWalkVariance))

        # Begin Sampling
        for iter in range(1, ns):

            # Sample latent positions
            for t in range(0, T):
                if t == 0:
                    logPosterior = conditionals.LogTime1ConditionalPosterior
                elif t == T - 1:
                    logPosterior = conditionals.LogTimeTConditionalPosterior
                else:
                    logPosterior = conditionals.LogMiddleTimeConditionalPosterior

                for i in range(0, n):
                    currentData["i"] = i
                    currentData["t"] = t
                    newPosition = MetropolisHastings(logPosterior, SampleFromIndMultivarNormalFixedVarP, positions[iter - 1, t, i], currentData)
                    positions[iter, t, i] = newPosition
                    currentData["X"][t, i] = newPosition
            
            # Sample radii using Metropolis-Hastings
            newRadii = MetropolisHastings(conditionals.LogRConditionalPosterior, SampleFromDirichlet, radii[iter - 1],
                                          currentData)
            radii[iter] = newRadii
            currentData["r"] = newRadii

            # Sample betaIN and betaOUT using Metropolis-Hastings
            newBetaIN = MetropolisHastings(conditionals.LogBetaINConditionalPosterior, SampleFromNormalFixedVar,
                                           betaIN[iter - 1], currentData)
            betaIN[iter] = newBetaIN
            currentData["betaIN"] = newBetaIN
            newBetaOUT = MetropolisHastings(conditionals.LogBetaOUTConditionalPosterior, SampleFromNormalFixedVar,
                                            betaOUT[iter - 1], currentData)
            betaOUT[iter] = newBetaOUT
            currentData["betaOUT"] = newBetaOUT

            # Sample tauSq and sigmaSq directly from conditional distribution
            newTauSq = conditionals.SampleTauSquared(currentData["X"])
            tauSq[iter] = newTauSq
            currentData["tauSq"] = newTauSq
            newSigmaSq = conditionals.SampleSigmaSquared(currentData["X"])
            sigmaSq[iter] = newSigmaSq
            currentData["sigmaSq"] = newSigmaSq
        
        return positions, radii, tauSq, sigmaSq, betaIN, betaOUT

def MetropolisHastings(ConditionalPosterior, ProposalSampler, currentValue, data, 
                       LogProposalEvaluate = None, proposalSymmetric = True, logPosterior = True):
    """
    Inputs:
    conditionalPosterior: one of the conditional posterior functions below
    proposalSampler : one of the proposal sampler functions below
    currentValue : the current value in the Markov chain of the parameter under study
    data : dictionary of all values needed by the conditionalPosterior, proposalSampler
    logProposalEvaluate : one of the "evaluate at" functions below (only needed for asymmetric proposal, default None)
    proposalSymmetric : is the proposal symmetric? (False for Normal, True for Dirichlet)
    logPosterior : are we passing in log functions everywhere? (defaults to True)

    Output:
        next value in the Markov chain (either proposalValue or currentValue)
    """
    proposalValue = ProposalSampler(currentValue)
    posteriorAtProposal = ConditionalPosterior(data, proposalValue)
    posteriorAtCurrent = ConditionalPosterior(data, currentValue)

    # If the proposal distribution is symmetric, we do not need to have a correction factor in the ratio
    if proposalSymmetric == True:
        if logPosterior == True:
            logAcceptanceRatio = posteriorAtProposal - posteriorAtCurrent
            acceptanceRatio = min(1, np.exp(logAcceptanceRatio))
        else:
            acceptanceRatio = min(1, posteriorAtProposal / posteriorAtCurrent)
    
    # If the proposal distribution is not symmetric, we need a correction factor
    else:
        # Evaluate proposal distribution each way
        logProposalValueGivenCurrent = LogProposalEvaluate(currentValue, proposalValue)
        logCurrentValueGivenProposal = LogProposalEvaluate(proposalValue, currentValue)

        if logPosterior == True:
            logAcceptanceRatio = (posteriorAtProposal + logCurrentValueGivenProposal) - (posteriorAtCurrent + logProposalValueGivenCurrent)
            acceptanceRatio = min(1, np.exp(logAcceptanceRatio))
        
        else:
            proposalValueGivenCurrent = np.exp(logProposalValueGivenCurrent)
            currentValueGivenProposal = np.exp(logCurrentValueGivenProposal)

            acceptanceRatio = min(1, (posteriorAtProposal * currentValueGivenProposal) / (posteriorAtCurrent * proposalValueGivenCurrent))
    
    # We have an acceptanceRatio. Now, we need to decide whether to accept or reject.
    if acceptanceRatio == 1:
        return proposalValue
    else:
        # Choose a random number between [0, 1]
        randomValue = np.random.rand()
        if randomValue < acceptanceRatio:
            return proposalValue
        else:
            return currentValue

def SampleFromIndMultivarNormal(mean, variance, dimension):
    cvMatrix = variance*np.eye(dimension)
    return np.random.multivariate_normal(mean, cvMatrix)

def SampleFromDirichlet(alphas):
    return np.random.dirichlet(alphas)