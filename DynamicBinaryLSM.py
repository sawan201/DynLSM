import numpy as np
from scipy.stats import invgamma

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
        pass  # To be implemented

    def SampleFromFullTauSquaredProposal(self, it: int):
        # fetch the latent positions at time 1 from our “currentState” snapshot
        X1 = self.currentState['X'][0]      # shape: (n, p)
        # pull in the prior hyperparameters for τ²
        theta = self.thetaTau               # θ_τ
        phi   = self.phiTau                 # φ_τ
        # compute the sum of squared norms of each actor’s position at t=1
        sum_sq = np.sum(X1 * X1)
        # form the posterior inverse‐gamma parameters
        post_shape = theta + 0.5 * self.n * self.p
        post_scale = phi   + 0.5 * sum_sq
        # draw a new τ² from IG(post_shape, post_scale)
        new_tau2 = invgamma.rvs(a=post_shape, scale=post_scale)
        # record this draw in our MCMC chains and in currentState
        self.tauSq[it]             = new_tau2
        self.currentState['tauSq'] = new_tau2

    def SampleFromFullSigmaSquaredProposal(self, it: int):
        # grab the entire latent‐position array (T, n, p)
        X = self.currentState['X']
        # pull in the prior hyperparameters for σ²
        theta = self.thetaSigma            # θ_σ
        phi   = self.phiSigma              # φ_σ
        # compute the frame‐to‐frame differences: X[t+1] − X[t]
        diffs = X[1:] - X[:-1]             # shape: (T−1, n, p)
        # sum of squared differences over t, i, p
        sum_sq = np.sum(diffs * diffs)
        # form the posterior inverse‐gamma parameters
        post_shape = theta + 0.5 * self.n * self.p * (self.T - 1)
        post_scale = phi   + 0.5 * sum_sq
        # draw a new σ² from IG(post_shape, post_scale)
        new_sigma2 = invgamma.rvs(a=post_shape, scale=post_scale)
        # record this draw in our chains and in currentState
        self.sigmaSq[it]              = new_sigma2
        self.currentState['sigmaSq']  = new_sigma2

    def ConditionalPosteriorBetaIN(self, beta_value: float) -> float:
        # log‐prior term for β_IN ~ N(ν_IN, η_IN)
        log_prior = -0.5 * (beta_value - self.nuIN)**2 / self.etaIN
        # pull current latent positions, radii, and β_OUT
        X       = self.currentState['X']         # (T, n, p)
        r       = self.currentState['r']         # (n,)
        betaOUT = self.currentState['betaOUT']   # scalar
        # build pairwise distances [T, n, n]
        diff      = X[:, :, None, :] - X[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        # compute affinities a_in = 1 - d / r_j, a_out = 1 - d / r_i
        a_in  = 1 - distances / r[np.newaxis, np.newaxis, :]
        a_out = 1 - distances / r[np.newaxis, :, None]
        # linear predictor η = β_IN·a_in + β_OUT·a_out
        eta = beta_value * a_in + betaOUT * a_out
        # log‐likelihood Σ y*η − log(1+exp(η))
        y       = self.data                     # (T, n, n)
        log_lik = np.sum(y * eta - np.logaddexp(0, eta))
        # return the sum of prior + likelihood
        return log_prior + log_lik

    def ConditionalPosteriorBetaOUT(self, beta_value: float) -> float:
        # log‐prior term for β_OUT ~ N(ν_OUT, η_OUT)
        log_prior = -0.5 * (beta_value - self.nuOUT)**2 / self.etaOUT
        # pull current latent positions, radii, and β_IN
        X      = self.currentState['X']        # (T, n, p)
        r      = self.currentState['r']        # (n,)
        betaIN = self.currentState['betaIN']   # scalar
        # build pairwise distances [T, n, n]
        diff      = X[:, :, None, :] - X[:, None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        # compute affinities
        a_in  = 1 - distances / r[np.newaxis, np.newaxis, :]
        a_out = 1 - distances / r[np.newaxis, :, None]
        # linear predictor η = β_IN·a_in + β_OUT·a_out
        eta = betaIN * a_in + beta_value * a_out
        # log‐likelihood Σ y*η − log(1+exp(η))
        y       = self.data
        log_lik = np.sum(y * eta - np.logaddexp(0, eta))
        # return the sum of prior + likelihood
        return log_prior + log_lik