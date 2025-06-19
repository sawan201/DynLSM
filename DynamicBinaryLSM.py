import numpy as np

class DynamicLSM():
    
    def __init__(self, data):
        """
        data is assumed to be a numpy array with shape (T, n, n)
        n is the number of actors
        T is the number of time periods
        """

        self.data = data
        self.n = data.shape[1]
        self.T = data.shape[0]

    def RunGibbs(self, numSteps, latentDim):
        """
        Gibbs sampling for Dynamic LSM.
        ns is the number of steps in the Markov Chain formed to estimate each parameter
        p is the dimension of the latent space we want to place our actors in
        """
        self.ns = numSteps
        self.p = latentDim

        # Set up empty Numpy arrays for each latent parameter
        self.positions = np.empty(shape=(self.ns, self.T, self.n, self.p))
        self.radii = np.empty(shape=(self.ns, self.n))
        self.tauSq = np.empty(shape=(self.ns))
        self.sigmaSq = np.empty(shape=(self.ns))
        self.betaIN = np.empty(shape=(self.ns))
        self.betaOUT = np.empty(shape=(self.ns))

    def MetroHastings(self, parameters):
        """
        Metropolis-Hastings sampling for Dynamic LSM.
        
        
        """
    def Conditional1(self, parameters):
        """
        Conditional sampling for Dynamic LSM.
        
        
        """
    
