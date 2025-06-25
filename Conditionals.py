# Function outline:

# def eta_ijt(beta_in, beta_out, r_i, r_j, X_it, X_jt):
#    return {eta_function}

# Continue here

import numpy as np
from scipy.stats import invgamma

class conditionals:
    def __init__(self,
                 theta_tau,  phi_tau,     #  Inv-Gamma prior for tau squared
                 theta_sig,  phi_sig,     #  Inv-Gamma prior for sigma squared
                 nu_in,      xi_in,       #  Normal prior for beta IN
                 nu_out,     xi_out):     #  Normal prior for beta OUT

        # hyper-parameters
        self.theta_tau, self.phi_tau = theta_tau, phi_tau
        self.theta_sig, self.phi_sig = theta_sig, phi_sig
        self.nu_in,   self.xi_in  = nu_in,  xi_in
        self.nu_out,  self.xi_out = nu_out, xi_out

        # log-likelihood for a SINGLE Bernoulli observation. Looping over will happen in the outer code. 
        # This can swap out for something like poisson over time, but the eta term will remain the same.
        self.log_p = lambda y, eta: y * eta - np.logaddexp(0.0, eta)  # y is the observed edge value, and eta is the linear predictor specified below
                                                                      # lamda is introducing an inline function that can take two arguements
                                                                      # logaddexp is the log transform we use, with the form np.logaddexp(a, b) = log(exp(a) + exp(b)). a = 0 since we are adding 1 as the first term in the expression.
                                                                    
    # Eta linear predictor for a SINGLE edge
    def eta(self, beta_in, beta_out, r_i, r_j, X_i, X_j):
        d_ijt = np.linalg.norm(X_i - X_j)          # Euclidean norm of x_i and x_j
        return (beta_in  * (1.0 - d_ijt / r_j) +   # toward j
                beta_out * (1.0 - d_ijt / r_i))    # from i

    # Log-likelihood of a SINGLE edge
    def log_p_ijt(self, y_ijt,
                  beta_in, beta_out,
                  r_i, r_j, X_i, X_j):
        eta_ijt = self.eta(beta_in, beta_out, r_i, r_j, X_i, X_j)
        return self.log_p(y_ijt, eta_ijt)

