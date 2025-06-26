import numpy as np                                        # import NumPy for numerical operations
from gibbs import RunBinaryGibbs                             # import the Gibbs sampler entry point
import initialize as init_mod                                # import initialization routines



def main():
    # initialize simulation and MCMC parameters
    T = 10                                                  # number of time periods
    n = 100                                                 # number of actors
    p = 2                                                   # latent-space dimensionality
    numSamples = 50000                                      # total MCMC samples
    burnin = 15000                                          # burn-in period for posterior summaries

    betaIN = 1.0                                            # true IN-effect parameter
    betaOUT = 2.0                                           # true OUT-effect parameter
    sigmaSq = 1.0 / (5 * n)**2                              # true variance of random walk
    mu = 0.02                                               # drift magnitude for influenced actors

    # specify prior hyperparameters
    nuIN, etaIN = betaIN, 100**2                            # Normal prior for betaIN
    nuOUT, etaOUT = betaOUT, 100**2                         # Normal prior for betaOUT
    thetaSigma, phiSigma = 9.0, 1.5                         # Inverse-gamma prior for sigmaSq
    alphas = np.ones(n)                                     # Dirichlet prior parameters for radii


if __name__ == '__main__':
    main()                                                   # execute main when script is run directly
