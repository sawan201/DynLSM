import numpy as np  # Import NumPy for numerical operations and array handling
import conditionalposteriors as cond  # Import the Conditionals module 
import gibbs as gibbs  # Import the Gibbs sampling module
import matplotlib.pyplot as plt
import os

class Simulation():
    def __init__(self, T, n, p, SigmaSq, TauSq, ThetaTau, ThetaSigma, PhiSigma, NuIn, XiIn, NuOut, XiOut, EtaIn, EtaOut, BetaIn, BetaOut, RandomWalkVariance, DirichletFactor, model_type, InitType, NumberOfSamples, BurnIn,
                 outPath = os.getcwd(), simName = ""):
        self.T = T  # Number of time points
        self.n = n  # Number of actors
        self.p = p  # Latent space dimensions
        self.SigmaSq = SigmaSq  # Step-size variance for latent position draws
        self.TauSq = TauSq  # Variance for the tau prior (used in the model)
        self.ThetaTau = ThetaTau  # Shape parameter for the tau prior (used in the model)
        self.ThetaSigma = ThetaSigma  # Shape parameter for the sigma prior (used in the model)
        self.PhiSigma = PhiSigma  # Scale parameter for the sigma prior (used in the model)
        self.NuIn = NuIn  # Input effect parameter for the model
        self.XiIn = XiIn  # Input effect parameter for the model
        self.NuOut = NuOut  # Output effect parameter for the model
        self.XiOut = XiOut  # Output effect parameter for the model
        self.EtaIn = EtaIn  # Input effect parameter for the model
        self.EtaOut = EtaOut  # Output effect parameter for the model
        self.BetaIn = BetaIn  # Input effect parameter for the model
        self.BetaOut = BetaOut  # Output effect parameter for the model
        self.RandomWalkVariance = RandomWalkVariance  # Variance for the random walk (used in the model)
        self.DirichletFactor = DirichletFactor  # Factor for the Dirichlet prior 
        self.model_type = model_type  # Type of model (e.g., "binary")
        self.InitType = InitType  # Initialization type for the model (e.g., "base")
        self.NumberOfSamples = NumberOfSamples  # Number of samples to draw in the Gibbs sampler
        self.BurnIn = BurnIn  # Number of initial samples to discard (burn-in period)
        self.outPath = outPath # Where to output the resulting .npz file to (a directory)
        self.simName = simName # Name to attach to the simulation file

    '''
    Helper Functions
    These functions are used to compute hyperparameters and perform operations needed for the simulation.
    '''

    def compute_phi_tau(self, X1, scale=1.05):  # Define a function to compute the phi_tau hyperparameter
        n, p = X1.shape  # Unpack the number of actors (n) and latent dimensions (p)
        sum_sq = np.sum(X1**2)  # Compute the sum of squared entries of the initial positions
        return scale * (sum_sq / (n * p))  # Scale by 1.05 (default) and normalize by n*p

    def scaled_inverse_norm(self, X1, i):  # Define a function to compute the scaled inverse norm for actor i
        n = X1.shape[0]  # Number of actors
        norms_inv = 1.0 / np.linalg.norm(X1, axis=1)  # Inverse of the Euclidean norm for each actor's vector
        return n * norms_inv[i] / np.max(norms_inv)  # Scale by n and normalize by the maximum inverse norm


    
    def sigmoid(self, x):
        # Prevent overflow in exp by clipping x
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

    '''
    Main Function
    '''

    def run(self, fixX = False, fixR = False, fixBetaIN = False, fixBetaOUT = False, 
            fixSigmaSq = False, fixTauSq = False):  
        '''
        Main function to run the simulation and sampling
        '''
        np.random.seed(181)
        T, n, p = self.T, self.n, self.p
        SigmaSq = self.SigmaSq
        TauSq = self.TauSq  # Variance for the tau prior
        SigmaSqRoot = np.sqrt(SigmaSq)  # Compute the square root of SigmaSq for sampling
        TauSqRoot = np.sqrt(TauSq)  # Compute the square root of TauSq for sampling
        model_type = self.model_type

        # Pre-allocate the latent positions array of shape (T, n, p)
        LargeX = np.zeros((T, n, p))

        # Simulate latent positions over time
        for t in range(T):  # Loop over each time point
            for i in range(n):  # Loop over each actor
                if t == 0:
                    mu = 0.0
                    LargeX[t, i] = np.random.normal(loc=mu, scale=TauSqRoot, size=p)  # Sample from N(0, SigmaSq) 
                else:
                    # For t > 0, use the previous time point's position as the mean
                    mu = LargeX[t-1, i]
                    LargeX[t, i] = np.random.normal(loc=mu, scale=SigmaSqRoot, size=p)  # Sample from N(mu, SigmaSq)

        # Extract the initial positions and compute phi_tau
        X1 = LargeX[0]  # Initial positions at time t=0
        phi_tau = self.compute_phi_tau(X1)  # Compute phi_tau using the helper function

        # Fit or sample from the conditional model using the hyperparameters
        P = cond.BinaryConditionals(
            theta_tau=self.ThetaTau,  # Shape parameter for tau^2 prior
            phi_tau=phi_tau,  # Scale parameter for tau^2 prior
            theta_sig=self.ThetaSigma,  # Shape parameter for sigma^2 prior
            phi_sig=self.PhiSigma,  # Scale parameter for sigma^2 prior
            nu_in=self.NuIn,  # Additional model hyperparameters
            xi_in=self.XiIn,
            nu_out=self.NuOut,
            xi_out=self.XiOut,
            alphas=None,  # Not used in this context, assuming None
            p=None
        )

        # Initialize an adjacency tensor Y of shape (T, n, n) with zeros
        Y = np.zeros((T, n, n), dtype=int)

        # Compute the r values by drawing from a Dirichlet with the desired distributions (added by LS, 7-10-25)
        rParameters = np.zeros(self.n)
        for i in range(self.n):
            rParameters[i] = self.scaled_inverse_norm(X1, i)
        trueR = np.random.dirichlet(rParameters)

        # Create the Y adjacency tensor by sampling edges
        for t in range(T):  # Loop over time points
            for i in range(n):  # Loop over actor i
                # r_i = self.scaled_inverse_norm(X1, i)  # Compute radius/influence for actor i
                # Use the r values computed at once above (changed by LS, 7-10-25)
                r_i = trueR[i]
                for j in range(n):  # Loop over actor j
                    if i == j:  # Skip self-edges
                        continue  # Continue to next j
                    
                    # r_j = self.scaled_inverse_norm(X1, j)  # Compute radius/influence for actor j
                    # Use the r values computed at once above (changed by LS, 7-10-25)
                    r_j = trueR[j]

                    eta = P.eta(
                        beta_in=self.BetaIn,  # Input effect parameter
                        beta_out=self.BetaOut,  # Output effect parameter
                        r_i=r_i,  # Radius for actor i
                        r_j=r_j,  # Radius for actor j
                        X_i=LargeX[t, i],  # Latent position of actor i at time t
                        X_j=LargeX[t, j]  # Latent position of actor j at time t
                    )
                    
                    # Convert log-odds to probability via the sigmoid function
                    prob = self.sigmoid(eta)  # Probability of an edge occurring

                    # Sample the edge as a Bernoulli trial
                    Y[t, i, j] = np.random.binomial(n=1, p=prob)  # Draw 0 or 1

        # ------------------------------------------------------------
        #  RUN GIBBS SAMPLER AND CHECK PARAMETER RECOVERY
        # ------------------------------------------------------------
        ns_total  = self.NumberOfSamples          # total MCMC sweeps
        burn_in   = self.BurnIn          # first draws to discard
        alphas    = np.ones(n)     # flat Dirichlet prior for radii

        sampler = gibbs.Gibbs(Y)   # create sampler object with data

        '''
        self, ns, p, modelType, initType, nuIN, xiIN, nuOUT, xiOUT, thetaSigma, phiSigma, 
                 thetaTau, phiTau, alphas, randomWalkVariance = 9, dirichletFactor = 200,
                 truth = None, fixX = False, fixR = False, fixBetaIN = False, fixBetaOUT = False, 
                 fixSigmaSq = False, fixTauSq = False)
        
        '''

        (X_Chain, R_Chain, tauSqChain, sigmaSqChain,
        betaIN_Chain, betaOUT_Chain) = sampler.RunGibbs(
            ns                 = ns_total,
            p                  = p,
            modelType          = self.model_type,
            initType           = self.InitType,
            nuIN               = self.NuIn,
            etaIN              = self.EtaIn,
            nuOUT              = self.NuOut,
            etaOUT             = self.EtaOut,
            thetaSigma         = self.ThetaSigma,
            phiSigma           = self.PhiSigma,
            thetaTau           = self.ThetaTau,
            phiTau             = phi_tau,
            alphas             = alphas,
            randomWalkVariance = self.RandomWalkVariance,
            dirichletFactor    = self.DirichletFactor,
            truth              = {"X" : LargeX, "R" : trueR, 
                                  "betaIN" : self.BetaIn, "betaOUT" : self.BetaOut, 
                                  "tauSq" : self.TauSq, "sigmaSq" : self.SigmaSq},
            fixX               = fixX,
            fixR               = fixR,
            fixBetaIN          = fixBetaIN,
            fixBetaOUT         = fixBetaOUT,
            fixSigmaSq         = fixSigmaSq,
            fixTauSq           = fixTauSq
        )

        # ---------- posterior summaries (after burn-in) ----------
        keep         = slice(burn_in, None)
        betaIN_hat   = betaIN_Chain[keep].mean()
        betaOUT_hat  = betaOUT_Chain[keep].mean()
        tauSq_hat    = tauSqChain[keep].mean()
        sigmaSq_hat  = sigmaSqChain[keep].mean()
        r_hat        = R_Chain[keep].mean(axis=0)           # actor-wise mean
        X_hat        = X_Chain[keep].mean(axis=0)           # latent positions

        # ---------- print comparison ----------
        print("\n=== PARAMETER RECOVERY CHECK ===")
        print(f"true  betaIN  = {self.BetaIn}   |  posterior mean = {betaIN_hat:6.3f}")
        print(f"true  betaOUT = {self.BetaOut}   |  posterior mean = {betaOUT_hat:6.3f}")
        print(f"true  SigmaSq = {SigmaSq:6.4f} |  posterior mean = {sigmaSq_hat:6.4f}")
        print(f"posterior mean of tauSq (true value varied each i) = {tauSq_hat:6.3f}")
        print("first five posterior means of r:", r_hat[:5])

        out_file = os.path.join(self.outPath, f"sim_run_{model_type}{self.simName}_ns{ns_total}_T{T}_n{n}_p{p}.npz")
        np.savez_compressed(out_file,
                            X_true=LargeX, Y=Y,
                            X_Chain=X_Chain, R_Chain=R_Chain,
                            betaIN_Chain=betaIN_Chain, betaOUT_Chain=betaOUT_Chain,
                            tauSqChain=tauSqChain, sigmaSqChain=sigmaSqChain,
                            trueX=LargeX, trueR=trueR, trueBetaIN=self.BetaIn,
                            trueBetaOUT=self.BetaOut, trueSigmaSq=self.SigmaSq,
                            trueTauSq=self.TauSq, thetaTau=self.ThetaTau, thetaSigma=self.ThetaSigma,
                            phiSigma=self.PhiSigma, nuIn=self.NuIn, xiIn=self.XiIn, nuOut=self.NuOut,
                            xiOut=self.XiOut, etaIn=self.EtaIn, etaOut=self.EtaOut,
                            randomWalkVariance=self.RandomWalkVariance)
        print(f"Saved full chains to {out_file}")