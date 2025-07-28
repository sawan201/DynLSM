import conditionalposteriors as cds
import initialize as init
import numpy as np
from scipy.stats import dirichlet
import time
from functools import wraps
from scipy.linalg import orthogonal_procrustes


class Gibbs:

    def __init__(self, Y):
        '''
        Creates an object of class Gibbs (that sampler can be run on multiple times)
        '''
        self.Y = Y
        self.T = Y.shape[0]
        self.n = Y.shape[1]

    '''
    Time recording decorator
    '''
    # Test comment
    def timer(func):
        @wraps(func)  # preserves the original function name and docstring
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"[TIMER] {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper

    def RunGibbs(self, ns, p, modelType, initType, nuIN, xiIN, nuOUT, xiOUT, thetaSigma, phiSigma, 
                 thetaTau, phiTau, alphas, 
                 betaRandomWalkVariance = 9, positionRandomWalkVariance = 0.09, dirichletFactor = 10000,
                 truth = None, fixX = False, fixR = False, fixBetaIN = False, fixBetaOUT = False, 
                 fixSigmaSq = False, fixTauSq = False):
        '''
        Inputs: 
            ns (int number of steps)
            p (int dimension of latent space)
            modelType (either "poisson" or "binary")
            initialization (either "base" or something else that we create later)
            nuIN (mean of prior on betaIN)
            xiIN (variance of prior on betaIN)
            nuOUT (mean of prior on betaOUT)
            xiOUT (variance of prior on betaOUT)
            thetaSigma (shape parameter of prior on SigmaSq)
            phiSigma (scale parameter of prior on SigmaSq)
            thetaTau (shape parameter of prior on TauSq)
            phiTau (scale parameter of prior on TauSq)
            alphas (parameters for Dirichlet prior on r_{1:n})
            betaRandomWalkVariance (variance of univariate normal random walk for betaIN and betaOUT proposal)
            positionRandomWalkVariance (variance of independent multivariate normal random walk for position proposals)
            dirichletFactor (factor to multiply prior iteration [sum to one] radii by in parameters of Dirichlet proposal)
            truth (if we are testing, a dictionary with the following keys:
                "X", "R", "betaIN", "betaOUT", "tauSq", "sigmaSq")
            fixX, fixR, fixBetaIN, fixBetaOUT, fixSigmaSq, fixTauSq:
                (Booleans to fix each parameter at the initial value for use in testing)
                
        Outputs:
            X (ns x T x n x p Numpy array of latent positions samples from Markov Chain)
            r (ns x n Numpy array of reach samples from Markov Chain)
            tauSq (ns array of tauSq samples from Markov Chain)
            sigmaSq (ns array of sigmaSq samples from Markov Chain)
            betaIN (ns array of betaIN samples from Markov Chain)
            betaOUT (ns array of betaOUT samples from Markov Chain)
        '''
        # Read in necessary parameters
        self.betaRandomWalkVariance = betaRandomWalkVariance
        self.positionRandomWalkVariance = positionRandomWalkVariance
        self.dirichletFactor = dirichletFactor

        # Assign the conditionals based on the input argument
        if modelType == "binary":
            conditionals = cds.BinaryConditionals(thetaTau, phiTau, thetaSigma, phiSigma, nuIN, xiIN,
                                                  nuOUT, xiOUT, alphas = alphas, p = p)
        elif modelType == "poisson":
            conditionals = cds.PoissonConditionals(thetaTau, phiTau, thetaSigma, phiSigma, nuIN, xiIN,
                                                   nuOUT, xiOUT, alphas = alphas, p = p)

        # Define key things:
        T = self.Y.shape[0]
        n = self.Y.shape[1]
        self.p = p

        # Initialize variables for acceptance ratios:
        self.positionAcceptances = 0
        self.radiiAcceptances = 0
        self.betaINAcceptances = 0
        self.betaOUTAcceptances = 0

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
        
        elif initType == "truth":
            initialization = init.InitializeToTruth(self.Y, positions, radii, betaIN, betaOUT, tauSq, sigmaSq,
                                                    truth["X"], truth["R"], truth["betaIN"], truth["betaOUT"],
                                                    truth["tauSq"], truth["sigmaSq"])

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
        print(self.currentData)

        # Begin Sampling
        for iter in range(1, ns):

            # For testing, we may want to fix positions.
            if fixX:
                positions[iter] = positions[iter - 1]
            # Otherwise, continue to sample X as normal.
            else:
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
                        newPosition = self.MetropolisHastings(logPosterior, self.SamplePositionsFromIndMultivarNormal, 
                                                              positions[iter - 1, t, i], self.currentData)
                        positions[iter, t, i] = newPosition
                        self.currentData["X"][t, i] = newPosition
                        # Add one to acceptances if we have a new position that is different from the prior one.
                        self.positionAcceptances = self.CheckAcceptance(newPosition, 
                                                                        positions[iter-1, t, i], 
                                                                        self.positionAcceptances)
                        print("Iteration", iter, "Time", t, "Actor", i, "completed.")

                # Procrustes after finishing the latent-position updates for this iteration
                X_mat   = positions[iter].reshape(T*n, p)   # Stack T time slices into one matrix
                X0      = positions[0].reshape(T*n, p)
                R, _    = orthogonal_procrustes(X_mat, X0)   # Solves the Procrustes problem
                X_rot = (X_mat @ R).reshape(T, n, p)   # Applying the rotation to every (i, t) coordinate, reshape gets original tensor form back

                positions[iter] = X_rot   # Store rotated positions
                self.currentData["X"] = X_rot.copy()

            # We may want to fix radii for testing.
            if fixR:
                radii[iter] = radii[iter - 1]
            # Otherwise, sample as normal.
            else:
                # Sample radii using Metropolis-Hastings
                newRadii = self.MetropolisHastings(conditionals.LogRConditionalPosterior, 
                                                   self.SampleFromDirichlet, radii[iter - 1],
                                                   self.currentData, 
                                                   LogProposalEvaluate = self.LogEvaluateDirichlet, 
                                                   proposalSymmetric= False)
                radii[iter] = newRadii
                self.currentData["r"] = newRadii
                # Add one to radiiAcceptances if we have a new chain value that is different from the one before.
                # We assume that if the first radius didn't change, they all didn't change.
                self.radiiAcceptances = self.CheckAcceptance(radii[iter, 0], radii[iter - 1, 0], self.radiiAcceptances)

            # We may want to fix betaIN via a keyword argument
            if fixBetaIN:
                betaIN[iter] = betaIN[iter - 1]
            # Otherwise, sample betaIN as normal.
            else:
                # Sample betaIN and betaOUT using Metropolis-Hastings
                newBetaIN = self.MetropolisHastings(conditionals.LogBetaINConditionalPosterior, 
                                                    self.SampleBetaFromNormalFixedVar,
                                                    betaIN[iter - 1], self.currentData)
                betaIN[iter] = newBetaIN
                self.currentData["betaIN"] = newBetaIN
                # Add one to betaINAcceptances if we have a new chain value that is different from the one before
                self.betaINAcceptances = self.CheckAcceptance(newBetaIN, betaIN[iter - 1], self.betaINAcceptances)

            # We may want to fix betaOUT via a keyword argument
            if fixBetaOUT:
                betaOUT[iter] = betaOUT[iter - 1]
            # Otherwise, sample betaOUT as normal.
            else:
                newBetaOUT = self.MetropolisHastings(conditionals.LogBetaOUTConditionalPosterior, 
                                                     self.SampleBetaFromNormalFixedVar,
                                                     betaOUT[iter - 1], self.currentData)
                betaOUT[iter] = newBetaOUT
                self.currentData["betaOUT"] = newBetaOUT
                # Add one to betaOUTAcceptances if we have a new chain value that is different from the one before
                self.betaOUTAcceptances = self.CheckAcceptance(newBetaOUT, betaOUT[iter - 1], self.betaOUTAcceptances)

            # We may want to fix tauSq via a keyword argument
            if fixTauSq:
                tauSq[iter] = tauSq[iter - 1]
            # Otherwise, sample tauSq as normal.
            else:
                # Sample tauSq and sigmaSq directly from conditional distribution
                newTauSq = conditionals.SampleTauSquared(self.currentData["X"])
                tauSq[iter] = newTauSq
                self.currentData["tauSq"] = newTauSq
            
            # We may want to fix sigmaSq via a keyword argument
            if fixSigmaSq:
                sigmaSq[iter] = sigmaSq[iter - 1]
            # Otherwise, sample sigmaSq as normal.
            else:
                newSigmaSq = conditionals.SampleSigmaSquared(self.currentData["X"])
                sigmaSq[iter] = newSigmaSq
                self.currentData["sigmaSq"] = newSigmaSq

            print("Iteration", iter, "Completed")
        
        # Calculate acceptance ratios
        self.positionAcceptanceRatio = self.positionAcceptances / (ns * T * n)
        self.radiiAcceptanceRatio = self.radiiAcceptances / ns
        self.betaINAcceptanceRatio = self.betaINAcceptances / ns
        self.betaOUTAcceptanceRatio = self.betaOUTAcceptances / ns

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
                acceptanceRatio = min(1, np.exp(np.clip(logAcceptanceRatio, -700, 700)))   # To avoid overflow errors

            else:
                acceptanceRatio = min(1, posteriorAtProposal / posteriorAtCurrent)
        
        # If the proposal distribution is not symmetric, we need a correction factor
        else:
            # Evaluate proposal distribution each way
            logProposalValueGivenCurrent = LogProposalEvaluate(currentValue, proposalValue)
            logCurrentValueGivenProposal = LogProposalEvaluate(proposalValue, currentValue)

            if logPosterior == True:
                logAcceptanceRatio = (posteriorAtProposal + logCurrentValueGivenProposal) - (posteriorAtCurrent + logProposalValueGivenCurrent)
                
                if logAcceptanceRatio >= 0:
                    acceptanceRatio = 1
                else:
                    acceptanceRatio = np.exp(logAcceptanceRatio)
                
            else:
                proposalValueGivenCurrent = np.exp(logProposalValueGivenCurrent)
                currentValueGivenProposal = np.exp(logCurrentValueGivenProposal)

                acceptanceRatio = min(1, (posteriorAtProposal * currentValueGivenProposal) / (posteriorAtCurrent * proposalValueGivenCurrent))
        
        # We have an acceptanceRatio. Now, we need to decide whether to accept or reject.
        if acceptanceRatio >= 1:
            return proposalValue
        else:
            # Choose a random number between [0, 1]
            randomValue = np.random.rand()
            if randomValue < acceptanceRatio:
                return proposalValue
            else:
                return currentValue

    # Samples from an independent multivariate normal distribution with self.randomWalkVariance and self.p dimensions
    def SamplePositionsFromIndMultivarNormal(self, mean):
        # Uses the self.p and self.randomWalkVariance attributes, only needs the mean
        cvMatrix = self.positionRandomWalkVariance*np.eye(self.p)
        return np.random.multivariate_normal(mean, cvMatrix)
    
    # Samples from a univariate normal distribution
    def SampleFromNormalFixedVar(self, mean):
        return np.random.normal(mean, np.sqrt(self.randomWalkVariance))

    # Samples proposal beta from univariate normal distribution with the specified beta
    def SampleBetaFromNormalFixedVar(self, mean):
        return np.random.normal(mean, np.sqrt(self.betaRandomWalkVariance))

    def SampleFromDirichlet(self, alphas):
        parameters = self.dirichletFactor * alphas
        return np.random.dirichlet(parameters)

    def LogEvaluateDirichlet(self, parameters, values):
        dirichletAdjustedParameters = self.dirichletFactor * parameters
        return dirichlet.logpdf(values, dirichletAdjustedParameters)

    # Function to determine whether things are equal, and if so, add one to the third parameter passed in
    def CheckAcceptance(self, value1, value2, numberAcceptances):
        if value1 == value2:
            return numberAcceptances + 1
        else:
            return numberAcceptances