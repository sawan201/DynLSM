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
            Full conditional log-posterior for beta_IN, as a function of the single argument beta_value.
            Uses current X’s, r’s, y’s and the Normal(nuIN, etaIN) prior stored on self.
            """
            # 1) log-prior: beta_IN ~ N(self.nuIN, self.etaIN)
            log_prior = -0.5 * (beta_value - self.nuIN)**2 / self.etaIN

            # 2) pull latent positions and radii from currentState
            X = self.currentState['X']    # shape (T, n, p)
            r = self.currentState['r']    # shape (n,)

            # 3) compute pairwise distances dist[t,i,j]
            diff = X[:, :, None, :] - X[:, None, :, :]  
            distances = np.linalg.norm(diff, axis=-1)  # shape (T, n, n)

            # 4) affinity a[t,i,j] = 1 – dist[t,i,j] / r[j]
            a = 1 - distances / r[np.newaxis, np.newaxis, :]

            # 5) linear predictor η[t,i,j] = beta_value * a[t,i,j]
            eta = beta_value * a

            # 6) log-likelihood: Σ[y[t,i,j]*η - log(1+exp(η))]
            y = self.data  # shape (T, n, n)
            log_lik = np.sum(y * eta - np.logaddexp(0, eta))

            # 7) return unnormalized log-posterior = log-prior + log-likelihood
            return log_prior + log_lik
