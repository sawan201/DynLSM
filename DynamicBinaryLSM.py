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
       

    def ConditionalPosteriorBetaIN(self, beta_value: float) -> float:
        """
        log f(beta_IN | rest)  ∝  log N(beta_IN|nuIN, etaIN) +
          Σ_{t,i,j} [ y_ijt * η_ijt  –  log(1+exp(η_ijt)) ]
        where η_ijt = beta_IN * a_in[t,i,j] + beta_OUT * a_out[t,i,j].
        """
        # 1) prior
        log_prior = -0.5 * (beta_value - self.nuIN)**2 / self.etaIN

        # 2) current state
        X = self.currentState['X']      # (T,n,p)
        r = self.currentState['r']      # (n,)
        betaOUT = self.currentState['betaOUT']

        # 3) pairwise distances d[t,i,j]
        diff = X[:, :, None, :] - X[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)  # (T,n,n)

        # 4) affinities
        #    a_in[t,i,j]  = 1 - d_{ijt}/r_j
        a_in = 1 - distances / r[np.newaxis, np.newaxis, :]
        #    a_out[t,i,j] = 1 - d_{ijt}/r_i
        a_out = 1 - distances / r[np.newaxis, :, None]

        # 5) linear predictor
        eta = beta_value * a_in + betaOUT * a_out

        # 6) log‐likelihood
        y = self.data
        log_lik = np.sum(y * eta - np.logaddexp(0, eta))

        return log_prior + log_lik

    def ConditionalPosteriorBetaOUT(self, beta_value: float) -> float:
        """
        Same as above, swapping roles of IN and OUT.
        """
        # 1) prior
        log_prior = -0.5 * (beta_value - self.nuOUT)**2 / self.etaOUT

        # 2) current state
        X = self.currentState['X']
        r = self.currentState['r']
        betaIN = self.currentState['betaIN']

        # 3) distances
        diff = X[:, :, None, :] - X[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        # 4) affinities
        a_in = 1 - distances / r[np.newaxis, np.newaxis, :]
        a_out = 1 - distances / r[np.newaxis, :, None]

        # 5) predictor
        eta = betaIN * a_in + beta_value * a_out

        # 6) log‐likelihood
        y = self.data
        log_lik = np.sum(y * eta - np.logaddexp(0, eta))

        return log_prior + log_lik