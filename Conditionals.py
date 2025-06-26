# Function outline:

# def eta_ijt(beta_in, beta_out, r_i, r_j, X_it, X_jt):
#    return {eta_function}

# Continue here

import numpy as np
from scipy.stats import invgamma

class conditionals:
    def __init__(self,
                 theta_tau,  phi_tau,   #  Inv-Gamma prior for tau squared
                 theta_sig,  phi_sig,   #  Inv-Gamma prior for sigma squared
                 nu_in,      xi_in,   #  Normal prior for beta IN
                 nu_out,     xi_out):   #  Normal prior for beta OUT

        # hyper-parameters
        self.theta_tau, self.phi_tau = theta_tau, phi_tau
        self.theta_sig, self.phi_sig = theta_sig, phi_sig
        self.nu_in,   self.xi_in  = nu_in,  xi_in
        self.nu_out,  self.xi_out = nu_out, xi_out

        # log-likelihood for a SINGLE Bernoulli observation. Looping over will happen in the outer code. 
        # This can swap out for something like poisson over time, but the eta term will remain the same.
        self.log_p = lambda y, eta: y * eta - np.logaddexp(0.0, eta)   # y is the observed edge value, and eta is the linear predictor specified below
                                                                       # lamda is introducing an inline function that can take two arguements
                                                                       # logaddexp is the log transform we use, with the form np.logaddexp(a, b) = log(exp(a) + exp(b)). a = 0 since we are adding 1 as the first term in the expression.
                                                                    
    # Eta linear predictor for a SINGLE edge
    def eta(self, beta_in, beta_out, r_i, r_j, X_i, X_j):
        d_ijt = np.linalg.norm(X_i - X_j)   # Euclidean norm of x_i and x_j
        return (beta_in  * (1.0 - d_ijt / r_j) +   # toward j
                beta_out * (1.0 - d_ijt / r_i))   # from i

    # Log-likelihood of a SINGLE edge
    def log_p_ijt(self, y_ijt,
                  beta_in, beta_out,
                  r_i, r_j, X_i, X_j):
        eta_ijt = self.eta(beta_in, beta_out, r_i, r_j, X_i, X_j)
        return self.log_p(y_ijt, eta_ijt)

    # Tau squared and sigma squared draws from inverse gamma distribution
    def sample_tau2(self, X_current):   # X_current is one snapshot of all latent positions at the present MCMC iteration                       
        X1 = X_current[0]  # This is taking the first slice of the X_current tensor, the first time point, which is what we want for tau squared
        n, p = X1.shape   # Giving us a tuple of of the dimensions of X1
        shape = self.theta_tau + 0.5 * n * p   # Specifying the shape parameter for the inverse gamma distribution
        scale = self.phi_tau  + 0.5 * np.sum(X1**2)   # This is the scale parameter for the inverse gamma distribution
        return invgamma.rvs(a=shape, scale=scale)   # Drawing a single time from the specific inverse gamma distribution 

    def sample_sigma2(self, X_current):
        diffs = X_current[1:] - X_current[:-1]   # Creating a tensor, where the t = 0 entry contains the difference between X2 - X1, the t = 1 entry contains the difference between X3 - X2, etc. This goes up to the XT - X(T-1) entry. 
        Tm1, n, p = diffs.shape   # Tm1 = T - 1, which is the number of increments
        a_post = self.theta_sig + 0.5 * n * p * Tm1   # Posterior shape parameter for the inverse gamma distribution
        b_post = self.phi_sig  + 0.5 * np.sum(diffs ** 2)   # Posterior scale parameter for the inverse gamma distribution
        return invgamma.rvs(a=a_post, scale=b_post)   # Drawing a single time from the specific inverse gamma distribution for sigma squared


