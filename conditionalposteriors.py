import numpy as np
from scipy.stats import invgamma
from scipy.special import gammaln
import line_profiler

class ConditionalPosteriors:
    def __init__(self,
                 theta_tau,  phi_tau,
                 theta_sig,  phi_sig,
                 nu_in,      xi_in,
                 nu_out,     xi_out,
                 alphas = None,
                 p = None):

        # hyper-parameters
        self.theta_tau, self.phi_tau = theta_tau, phi_tau
        self.theta_sig, self.phi_sig = theta_sig, phi_sig
        self.nu_in,   self.xi_in  = nu_in,  xi_in
        self.nu_out,  self.xi_out = nu_out, xi_out
        self.alphas = alphas
        self.p = p

        # log-likelihood for a SINGLE BERNOULLI observation
        # This can swap out for something like poisson over time, but the eta term will remain the same.
        self.log_p = lambda y, eta: y * eta - np.logaddexp(0.0, eta)   # y is the observed edge value, and eta is the linear predictor specified below
                                                                       # lamda is introducing an inline function that can take two arguements
                                                                       # logaddexp is the log transform we use, with the form np.logaddexp(a, b) = log(exp(a) + exp(b)). a = 0 since we are adding 1 as the first term in the expression.
    @line_profiler.profile                                 
    def LogLikelihood(self, Y, X, r, betaIN, betaOUT, tauSq, sigmaSq):
        T, n, _ = Y.shape   # Last dimension is n, ignore with _
        log_like = 0.0
        # WRITE A DISTANCE MATRIX FUNCTION TO CALCULATE DISTANCES BETWEEN ALL PAIRS OF ACTORS AT EACH TIME POINT, SYMMETRIC SO WE DO NOT HAVE TO DO IT TWICE
        for t in range(T):
            for i in range(n):   # Going over every sender
                for j in range(n):   # Every reciever
                    if i == j:
                        continue   # Skip self-edges
                    
                    log_like += self.LogPijt(
                        Y, X, r, betaIN, betaOUT, tauSq, sigmaSq, i, j, t
                    )

        return log_like


    #######################
    # VARIANCE PARAMETERS #
    #######################

    #Why are the "shape/scale" parameters called "a_post" and "b_post" in SampleSigmaSquared but not SampleTauSquared?
    @line_profiler.profile                                 
    def SampleTauSquared(self, X_current):   # X_current is one snapshot of all latent positions at the present MCMC iteration                       
        X1 = X_current[0]  # This is taking the first slice of the X_current tensor, the first time point, which is what we want for tau squared
        n, p = X1.shape   # Giving us a tuple of of the dimensions of X1
        shape = self.theta_tau + 0.5 * n * p   # Specifying the shape parameter for the inverse gamma distribution
        scale = self.phi_tau  + 0.5 * np.sum(X1**2)   # This is the scale parameter for the inverse gamma distribution
        return invgamma.rvs(a=shape, scale=scale)   # Drawing a single time from the specific inverse gamma distribution 
    @line_profiler.profile                                 
    def SampleSigmaSquared(self, X_current):
        diffs = X_current[1:] - X_current[:-1]   # Creating a tensor, where the t = 0 entry contains the difference between X2 - X1, the t = 1 entry contains the difference between X3 - X2, etc. This goes up to the XT - X(T-1) entry. 
        Tm1, n, p = diffs.shape   # Tm1 = T - 1, which is the number of increments
        a_post = self.theta_sig + 0.5 * n * p * Tm1   # Posterior shape parameter for the inverse gamma distribution
        b_post = self.phi_sig  + 0.5 * np.sum(diffs ** 2)   # Posterior scale parameter for the inverse gamma distribution
        return invgamma.rvs(a=a_post, scale=b_post)   # Drawing a single time from the specific inverse gamma distribution for sigma squared


    ##############
    # Log-priors #
    ##############

    ### BETA LOG-PRIORS ###
    @line_profiler.profile                                 
    def LogBetaINPrior(self, beta_val):
        return -0.5 * (beta_val - self.nu_in)  ** 2 / self.xi_in
    @line_profiler.profile                                 
    def LogBetaOUTPrior(self, beta_val):
        return -0.5 * (beta_val - self.nu_out) ** 2 / self.xi_out
                                  
    ### LATENT POSITION LOG-PRIORS ###
    LOG2PI = np.log(2.0 * np.pi)   # Pre-storing log(2pi) so it doesn't have to be recalculated every time in the loop
    
    @line_profiler.profile                                 
    def mvnorm_logpdf(self, x, mu, var):   # Helper function to calculate the log density of a multivariate normal distribution
        diff = x - mu   # Performing element wise subtraction to get the deviation vector
        p = diff.size   # Counting all elements in the array
        return -0.5 * (p * self.LOG2PI + p * np.log(var) + diff.dot(diff) / var)   # Evaluating formula for multivariate normal with covariance sigma squared * I_p
    @line_profiler.profile                                 
    def LogX1Prior(self, X, tauSq, sigmaSq, i):   # Returning the log prior contribution from one actor i 
        x1 = X[0, i]   # Taking out the first time slice vector for actor i
        x2 = X[1, i]   # Taking out the second time slide vector for actor i
        lp = self.mvnorm_logpdf(x1, np.zeros_like(x1), tauSq)   # Using np.zeros_like(x1) to create a zero vector of the same shape as x1, which is the mean of the prior distribution
        lp += self.mvnorm_logpdf(x2, x1, sigmaSq)   # This is the log prior contribution of the second time slice from the initial latent positions distribution
        return lp   
    @line_profiler.profile                                 
    def LogMiddleXPrior(self, X, sigmaSq, i, t):
        xm1 = X[t - 1, i]   # Taking out the t-1 timeslice vector for actor i
        x   = X[t, i]   # Taking out the t timeslice vector for actor i
        xp1 = X[t + 1, i]   # Taking out the t+1 timeslice vector for actor i
        lp  = self.mvnorm_logpdf(x, xm1, sigmaSq)
        lp += self.mvnorm_logpdf(xp1, x, sigmaSq)
        return lp
    @line_profiler.profile                                 
    def LogXTPrior(self, X, sigmaSq, i):   # Returning the log prior contribution from the last actor i
        xm1 = X[-2, i]   # Taking out the second to last time slice vector for actor i
        x   = X[-1, i]   # Taking out the last time slice vector for actor i
        return self.mvnorm_logpdf(x, xm1, sigmaSq)
    @line_profiler.profile                                 
    def LogRPrior(self, r):
        return 0.0   # Any constant works, so returning 0.0 is conventional here


    ##############
    # Posteriors #
    ##############
    @line_profiler.profile                                 
    def LogTime1ConditionalPosterior(self, currentData, xValue):
        i, t = currentData["i"], currentData["t"]   # t should be 0 here, done to identify which element of latent postion tensor we update
        X_prop = currentData["X"].copy()   # Creating copy of current latent positions tensor to temporarily modify without affecting current state of chain
        X_prop[t, i] = xValue   # Replacing old xi1 with new proposal

        ll = self.LogLikelihood(currentData["Y"], X_prop,
                                currentData["r"],
                                currentData["betaIN"],
                                currentData["betaOUT"],
                                currentData["tauSq"],
                                currentData["sigmaSq"])   # Evaluating the log likelihood of observed network Y under proposed X

        lp = self.LogX1Prior(X_prop,
                             currentData["tauSq"],
                             currentData["sigmaSq"],
                             i)   # Log-prior contribution from xi1

        return ll + lp   # Log posterior
    @line_profiler.profile
    def LogMiddleTimeConditionalPosterior(self, currentData, xValue):
        i, t = currentData["i"], currentData["t"]   # Identifying which element of latent position tensor is updated
        X_prop = currentData["X"].copy()
        X_prop[t, i] = xValue

        ll = self.LogLikelihood(currentData["Y"], X_prop,
                                currentData["r"],
                                currentData["betaIN"],
                                currentData["betaOUT"],
                                currentData["tauSq"],
                                currentData["sigmaSq"])
        lp = self.LogMiddleXPrior(X_prop,
                                  currentData["sigmaSq"],
                                  i, t)

        return ll + lp
    @line_profiler.profile
    def LogTimeTConditionalPosterior(self, currentData, xValue):
        i, t = currentData["i"], currentData["t"]
        X_prop = currentData["X"].copy()
        X_prop[t, i] = xValue

        ll = self.LogLikelihood(currentData["Y"], X_prop,
                                currentData["r"],
                                currentData["betaIN"],
                                currentData["betaOUT"],
                                currentData["tauSq"],
                                currentData["sigmaSq"])

        lp = self.LogXTPrior(X_prop,
                             currentData["sigmaSq"],
                             i)

        return ll + lp
    @line_profiler.profile
    def LogRConditionalPosterior(self, currentData, rValues):
        ll = self.LogLikelihood(currentData["Y"],
                                currentData["X"],
                                rValues,
                                currentData["betaIN"],
                                currentData["betaOUT"],
                                currentData["tauSq"],
                                currentData["sigmaSq"])

        lp = self.LogRPrior(rValues)
        return ll + lp
    @line_profiler.profile
    def LogBetaINConditionalPosterior(self, currentData, betaINValue):
        ll = self.LogLikelihood(currentData["Y"],
                                currentData["X"],
                                currentData["r"],
                                betaINValue,
                                currentData["betaOUT"],
                                currentData["tauSq"],
                                currentData["sigmaSq"])

        lp = self.LogBetaINPrior(betaINValue)
        return ll + lp
    @line_profiler.profile
    def LogBetaOUTConditionalPosterior(self, currentData, betaOUTValue):
        ll = self.LogLikelihood(currentData["Y"],
                                currentData["X"],
                                currentData["r"],
                                currentData["betaIN"],
                                betaOUTValue,
                                currentData["tauSq"],
                                currentData["sigmaSq"])

        lp = self.LogBetaOUTPrior(betaOUTValue)
        return ll + lp




### IMPLEMENTING SUBCLASSES FOR MODULAR SWAPPING ###
class BinaryConditionals(ConditionalPosteriors):
    # Eta linear predictor for a SINGLE edge
    @line_profiler.profile                                 
    def eta(self, beta_in, beta_out, r_i, r_j, X_i, X_j):
        d_ijt = np.linalg.norm(X_i - X_j)   # Euclidean norm of x_i and x_j
        return (beta_in  * (1.0 - d_ijt / r_j) +   # toward j
                beta_out * (1.0 - d_ijt / r_i))   # from i

    # Log-likelihood of a SINGLE edge
    @line_profiler.profile                                 
    def LogPijt(self, Y, X, r,
                  betaIN, betaOUT,
                  tauSq, sigmaSq,
                  i, j, t):
        y = Y[t, i, j]
        eta_ijt = self.eta(betaIN, betaOUT, r[i], r[j], X[t, i], X[t, j])
        return self.log_p(y, eta_ijt)

### CHECK OVER, GENERATED FROM CHAT ###
class PoissonConditionals(ConditionalPosteriors):
    """
    Dynamic latent–space model with Poisson edges
    Y_ijt  ~  Poisson( λ_ijt ),     log λ_ijt = eta_ijt
    """

    def __init__(self,
                 theta_tau,  phi_tau,
                 theta_sig,  phi_sig,
                 nu_in,      xi_in,
                 nu_out,     xi_out,
                 alphas=None,
                 p=None):

        super().__init__(theta_tau, phi_tau,
                         theta_sig,  phi_sig,
                         nu_in,      xi_in,
                         nu_out,     xi_out,
                         alphas=alphas,
                         p=p)

        # overwrite the Bernoulli log-pmf with the Poisson one
        #   log p(y|η) = y*η − exp(η) − log(y!)
        self.log_p = lambda y, eta: y * eta - np.exp(eta) - gammaln(y + 1)

    # — same η form as the binary model —
    def eta(self, beta_in, beta_out, r_i, r_j, X_i, X_j):
        d_ijt = np.linalg.norm(X_i - X_j)
        return (beta_in  * (1.0 - d_ijt / r_j) +
                beta_out * (1.0 - d_ijt / r_i))

    # log-likelihood contribution of **one** dyad
    def LogPijt(self, Y, X, r,
                betaIN, betaOUT,
                tauSq, sigmaSq,
                i, j, t):

        y      = Y[t, i, j]
        eta_ij = self.eta(betaIN, betaOUT, r[i], r[j], X[t, i], X[t, j])
        return self.log_p(y, eta_ij)















# END OF CODE
