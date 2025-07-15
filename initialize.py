import numpy as np

class AbstractInitialization:

    def __init__(self, Y, X, r, betaIN, betaOUT, tauSq, sigmaSq):
        # Expects that the following are passed in:
        self.Y = Y  # (T, n, n) Numpy tensor representing adjacency matrices
        self.X = X  # empty (ns, T, n, p) Numpy tensor representing latent positions
        self.r = r  # empty (ns, n) Numpy matrix representing social radii
        self.betaIN = betaIN    # empty (ns) Numpy array representing betaIN values
        self.betaOUT = betaOUT  # empty (ns) Numpy array representing betaOUT values
        self.tauSq = tauSq      # empty (ns) Numpy array representing tauSq values
        self.sigmaSq = sigmaSq  # empty (ns) Numpy array representing sigmaSq values
        self.T = self.X.shape[1]    # number of time periods in dataset
        self.n = self.X.shape[2]    # number of actors
        self.p = self.X.shape[3]    # dimension of latent space

    def InitializeAll(self):
        self.InitializeX()
        self.InitializeR()
        self.InitializeBetaIN()
        self.InitializeBetaOUT()
        self.InitializeSigmaSq()
        self.InitializeTauSq()
        return self.X, self.r, self.betaIN, self.betaOUT, self.tauSq, self.sigmaSq

    def InitializeX(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeR(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeTauSq(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeSigmaSq(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeBetaIN(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeBetaOUT(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

class InitializeToTruth(AbstractInitialization):
    # Allows us to initialize to the true values for testing
    def __init__(self, Y, X, r, betaIN, betaOUT, tauSq, sigmaSq, 
                 trueX, trueR, trueBetaIN, trueBetaOUT, trueTauSq, trueSigmaSq):
        self.Y = Y
        self.X = X
        self.r = r
        self.betaIN = betaIN
        self.betaOUT = betaOUT
        self.tauSq = tauSq
        self.sigmaSq = sigmaSq
        self.trueX = trueX
        self.trueR = trueR
        self.trueBetaIN = trueBetaIN
        self.trueBetaOUT = trueBetaOUT
        self.trueTauSq = trueTauSq
        self.trueSigmaSq = trueSigmaSq
    
    def InitializeX(self):
        self.X[0] = self.trueX
    
    def InitializeR(self):
        self.r[0] = self.trueR
    
    def InitializeBetaIN(self):
        self.betaIN[0] = self.trueBetaIN
    
    def InitializeBetaOUT(self):
        self.betaOUT[0] = self.trueBetaOUT
    
    def InitializeSigmaSq(self):
        self.sigmaSq[0] = self.trueSigmaSq
    
    def InitializeTauSq(self):
        self.tauSq[0] = self.trueTauSq

class BaseInitialization(AbstractInitialization):
    # uses the constructor from AbstractInitialization

    def InitializeX(self):
        # Sets the first step (X[0]) to a T x n x p array of zeros. (starts latent positions at origin)
        self.X[0] = np.zeros(shape=(self.T, self.n, self.p))

    def InitializeR(self):
        # Sets the first step for R to 1/n at every point
        self.r[0] = (1/self.n)*np.ones(shape=(self.n))
    
    def InitializeBetaIN(self):
        # Initializes to arbitrary value
        self.betaIN[0] = 0
    
    def InitializeBetaOUT(self):
        # Initializes to arbitrary value
        self.betaOUT[0] = 0
    
    def InitializeSigmaSq(self):
        # Initializes to arbitrary value
        self.sigmaSq[0] = 9
    
    def InitializeTauSq(self):
        # Initializes to arbitrary value
        self.tauSq[0] = 9

class ImprovedInitialization(AbstractInitialization):
    def InitializeR(self):
        # Initialization as done in paper
        # Get the total interactions between two points across all time periods
        sumAcrossT = np.sum(self.Y, axis=0)
        # Sum to create one vector representing total incoming and another representing total outgoing
        sumAcrossSecondAxis = np.sum(sumAcrossT, axis=0)
        sumAcrossThirdAxis = np.sum(sumAcrossT, axis=1)
        # Find the number incoming + number outgoing for each
        totalInteractions = sumAcrossSecondAxis + sumAcrossThirdAxis
        # Ensure that this sums to one (every interaction is represented twice)
        self.r = totalInteractions / 2 * np.sum(totalInteractions)