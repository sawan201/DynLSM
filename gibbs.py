import conditionalposteriors as cds
import initialize as init
import numpy as np
from scipy.stats import dirichlet

class Gibbs:

    def __init__(self, Y):
        '''
        Creates an object of class Gibbs (that sampler can be run on multiple times)
        '''
        self.Y = Y
        self.T = Y.shape[0]
        self.n = Y.shape[1]

    def RunGibbs(self, ns, p, modelType, initType, nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                 thetaTau, phiTau, alphas, randomWalkVariance = 9):
            '''
            Inputs: 
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
            # Read in necessary parameters
            self.randomWalkVariance = randomWalkVariance

            # Assign the conditionals based on the input argument
            if modelType == "binary":
                conditionals = cds.BinaryConditionals(nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                                                    thetaTau, phiTau, alphas, n, p, T)
            elif modelType == "poisson":
                conditionals = cds.PoissonConditionals(nuIN, etaIN, nuOUT, etaOUT, thetaSigma, phiSigma, 
                                                    thetaTau, phiTau, alphas, n, p, T)

            # Define key things:
            T = self.Y.shape[0]
            n = self.Y.shape[1]
            self.p = p

            # Set up empty Numpy arrays
            positions = np.empty(shape=(ns, T, n, p))
            radii = np.empty(shape=(ns, n))
            betaIN = np.empty(shape=(ns))
            betaOUT = np.empty(shape=(ns))
            tauSq = np.empty(shape=(ns))
            sigmaSq = np.empty(shape=(ns))

            # Assign the initialization based on the input argument
            if initType == "base":
                initialization = init.BaseInitialization(self.Y, positions, radii, betaIN, betaOUT, tauSq, sigmaSq)
            
            # Set initial values for all
            positions, radii, betaIN, betaOUT, tauSq, sigmaSq = initialization.InitializeAll()

            # Setup the currentData dictionary
            self.currentData = {"Y" : self.Y, 
                                "X" : positions[0].copy(),
                                "r" : radii[0].copy(),
                                "betaIN" : betaIN[0],
                                "betaOUT" : betaOUT[0],
                                "tauSq" : tauSq[0],
                                "sigmaSq" : sigmaSq[0]}            

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
                        self.currentData["i"] = i
                        self.currentData["t"] = t
                        newPosition = self.MetropolisHastings(logPosterior, self.SampleFromIndMultivarNormal, positions[iter - 1, t, i], self.currentData)
                        positions[iter, t, i] = newPosition
                        self.currentData["X"][t, i] = newPosition
                
                # Sample radii using Metropolis-Hastings
                newRadii = self.MetropolisHastings(conditionals.LogRConditionalPosterior, self.SampleFromDirichlet, radii[iter - 1],
                                                   self.currentData, 
                                                   LogProposalEvaluate = self.LogEvaluateDirichlet, 
                                                   proposalSymmetric= False)
                radii[iter] = newRadii
                self.currentData["r"] = newRadii

                # Sample betaIN and betaOUT using Metropolis-Hastings
                newBetaIN = self.MetropolisHastings(conditionals.LogBetaINConditionalPosterior, self.SampleFromNormalFixedVar,
                                            betaIN[iter - 1], self.currentData)
                betaIN[iter] = newBetaIN
                self.currentData["betaIN"] = newBetaIN
                newBetaOUT = self.MetropolisHastings(conditionals.LogBetaOUTConditionalPosterior, self.SampleFromNormalFixedVar,
                                                betaOUT[iter - 1], self.currentData)
                betaOUT[iter] = newBetaOUT
                self.currentData["betaOUT"] = newBetaOUT

                # Sample tauSq and sigmaSq directly from conditional distribution
                newTauSq = conditionals.SampleTauSquared(self.currentData["X"])
                tauSq[iter] = newTauSq
                self.currentData["tauSq"] = newTauSq
                newSigmaSq = conditionals.SampleSigmaSquared(self.currentData["X"])
                sigmaSq[iter] = newSigmaSq
                self.currentData["sigmaSq"] = newSigmaSq
            
            return positions, radii, tauSq, sigmaSq, betaIN, betaOUT

    def MetropolisHastings(self, ConditionalPosterior, ProposalSampler, currentValue, data, 
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

    # Samples from an independent multivariate normal distribution with self.randomWalkVariance and self.p dimensions
    def SampleFromIndMultivarNormal(self, mean):
        # Uses the self.p and self.randomWalkVariance attributes, only needs the mean
        cvMatrix = self.randomWalkVariance*np.eye(self.p)
        return np.random.multivariate_normal(mean, cvMatrix)
    
    # Samples from a univariate normal distribution
    def SampleFromNormalFixedVar(self, mean):
        return np.random.normal(mean, np.sqrt(self.randomWalkVariance))

    def SampleFromDirichlet(self, alphas):
        return np.random.dirichlet(alphas)

    def LogEvaluateDirichlet(self, parameters, values):
        return dirichlet.logpdf(values, parameters)