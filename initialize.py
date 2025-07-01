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

    def InitializeX():
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeR():
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeTauSq():
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeSigmaSq():
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeBetaIN():
        raise NotImplementedError("This method must be implemented by a subclass.")

    def InitializeBetaOUT():
        raise NotImplementedError("This method must be implemented by a subclass.")

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
        self.betaIN[0] = 1
    
    def InitializeBetaOUT(self):
        # Initializes to arbitrary value
        self.betaOUT[0] = 1
    
    def InitializeSigmaSq(self):
        # Initializes to arbitrary value
        self.sigmaSq[0] = 9
    
    def InitializeTauSq(self):
        # Initializes to arbitrary value
        self.tauSq[0] = 9