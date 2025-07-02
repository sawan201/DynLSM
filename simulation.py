import numpy as np  # Import NumPy for numerical operations and array handling
import conditionalposteriors as cond  # Import the Conditionals module 
import gibbs as gibbs  # Import the Gibbs sampling module
import matplotlib.pyplot as plt

'''
Helper Functions
These functions are used to compute hyperparameters and perform operations needed for the simulation.
'''

def compute_phi_tau(X1, scale=1.05):  # Define a function to compute the phi_tau hyperparameter
    n, p = X1.shape  # Unpack the number of actors (n) and latent dimensions (p)
    sum_sq = np.sum(X1**2)  # Compute the sum of squared entries of the initial positions
    return scale * (sum_sq / (n * p))  # Scale by 1.05 (default) and normalize by n*p

def scaled_inverse_norm(X1, i):  # Define a function to compute the scaled inverse norm for actor i
    n = X1.shape[0]  # Number of actors
    norms_inv = 1.0 / np.linalg.norm(X1, axis=1)  # Inverse of the Euclidean norm for each actor's vector
    return n * norms_inv[i] / np.max(norms_inv)  # Scale by n and normalize by the maximum inverse norm

def sigmoid(x):  # Define the sigmoid (logistic) function to squash values into [0, 1]
    return 1.0 / (1.0 + np.exp(-x))  # Compute 1 / (1 + exp(-x))

'''
Main Function
'''

np.random.seed(42)

def main():  # Main function to run the simulation and sampling
    T, n, p = 6, 30, 2  # Set time points (T), number of actors (n), and latent space dimensions (p)
    SigmaSq = 1.0 / (5 * n)**2  # Compute the step-size variance for latent position draws

    # Pre-allocate the latent positions array of shape (T, n, p)
    LargeX = np.zeros((T, n, p))

    # Simulate latent positions over time
    for t in range(T):  # Loop over each time point
        for i in range(n):  # Loop over each actor
            mu = 0.0 if t == 0 else LargeX[t-1, i]  # If t=0, center at origin; otherwise use last position
            LargeX[t, i] = np.random.normal(loc=mu, scale=SigmaSq, size=p)  # Sample from N(mu, SigmaSq)

    # Extract the initial positions and compute phi_tau
    X1 = LargeX[0]  # Initial positions at time t=0
    phi_tau = compute_phi_tau(X1)  # Compute phi_tau using the helper function
   

    # Fit or sample from the conditional model using the hyperparameters
    P = cond.BinaryConditionals(
        theta_tau=2.05,  # Shape parameter for tau^2 prior
        phi_tau=phi_tau,  # Scale parameter for tau^2 prior
        theta_sig=9.0,  # Shape parameter for sigma^2 prior
        phi_sig=1.5,  # Scale parameter for sigma^2 prior
        nu_in=0.0,  # Additional model hyperparameters
        xi_in=1.0,
        nu_out=0.0,
        xi_out=1.0,
        alphas=None,  # Not used in this context, assuming None
        p=None
    )
    
    # Initialize an adjacency tensor Y of shape (T, n, n) with zeros
    Y = np.zeros((T, n, n), dtype=int)

    # Create the Y adjacency tensor by sampling edges
    for t in range(T):  # Loop over time points
        for i in range(n):  # Loop over actor i
            r_i = scaled_inverse_norm(X1, i)  # Compute radius/influence for actor i
            for j in range(n):  # Loop over actor j
                if i == j:  # Skip self-edges
                    continue  # Continue to next j

                r_j = scaled_inverse_norm(X1, j)  # Compute radius/influence for actor j

                
                eta = P.eta(
                    beta_in=1.0,  # Input effect parameter
                    beta_out=2.0,  # Output effect parameter
                    r_i=r_i,  # Radius for actor i
                    r_j=r_j,  # Radius for actor j
                    X_i=LargeX[t, i],  # Latent position of actor i at time t
                    X_j=LargeX[t, j]  # Latent position of actor j at time t
                )

                

                # Convert log-odds to probability via the sigmoid function
                prob = sigmoid(eta)  # Probability of an edge occurring

                # Sample the edge as a Bernoulli trial
                Y[t, i, j] = np.random.binomial(n=1, p=prob)  # Draw 0 or 1

   

    '''
     X_new, R_new, tauSq_new, sigmaSq_new, betaIN_new, betaOUT_new = gibbs.RunBinaryGibbs(
        Y=Y,  # The sampled adjacency tensor
        ns=n,
        p=2,
        modelType="binary",  # Model type for Gibbs sampling
        initType="base",  # Initialization type for Gibbs sampling
        nuIN=0.0,
        etaIN=1.0,
        nuOUT=0.0,
        etaOUT=1.0,
        thetaSigma=9.0,  # Shape parameter for sigma^2 prior
        phiSigma=1.5,  # Scale parameter for sigma^2 prior
        thetaTau=2.05,  # Shape parameter for tau^2 prior
        phiTau=phi_tau,  # Scale parameter for tau^2 prior
        alphas=None,  # Not sure what to put here, assuming None for now   
        randomWalkVariance=9.0  # Variance for the random walk proposal
    )
    print("Gibbs sampling results:")  # Label the output
    print("X_new:", X_new)  # Display the sampled latent positions
    print("R_new:", R_new)  # Display the sampled reach (radii)     
    print("tauSq_new:", tauSq_new)  # Display the sampled tau^2
    print("sigmaSq_new:", sigmaSq_new)  # Display the sampled sigma^2       
    print("betaIN_new:", betaIN_new)  # Display the sampled beta_IN
    print("betaOUT_new:", betaOUT_new)  # Display the sampled beta_OUT
    # Note: The alphas parameter is not used in the Gibbs sampling call,


    #The rest of the program may not work right now, as it is not fully implemented.

    betaIN_ = (1.0/n) * np.sum(betaIN_new)
    betaOUT_ = (1.0/n) * np.sum(betaOUT_new)    
    r_ = (1.0/n) * np.sum(R_new, axis=0)  # Average radius across actors
    tauSq_ = (1.0/n) * np.sum(tauSq_new)  # Average tau^2 across actors
    sigmaSq_ = (1.0/n) * np.sum(sigmaSq_new)  # Average sigma^2 across actors   

    
    '''

    print(f"Big X: {LargeX[:10]}")  # Print the first 10 time points of latent positions
    print(f"Adjacency tensor Y: {Y[:10]}")  # Print the sampled adjacency tensor

    plt.plot(LargeX[:, 0, 1])
    plt.show()










'''

    # ------------------------------------------------------------
    #  RUN GIBBS SAMPLER AND CHECK PARAMETER RECOVERY
    # ------------------------------------------------------------
    ns_total  = 7_000          # total MCMC sweeps
    burn_in   = 1_000          # first draws to discard
    alphas    = np.ones(n)     # flat Dirichlet prior for radii

    sampler = gibbs.Gibbs(Y)   # create sampler object with data

    X_chain, R_chain, tauSq_chain, sigmaSq_chain, \
    betaIN_chain, betaOUT_chain = sampler.RunGibbs(
        ns                 = ns_total,
        p                  = p,
        modelType          = "binary",
        initType           = "base",
        nuIN               = 0.0,
        etaIN              = 1.0,
        nuOUT              = 0.0,
        etaOUT             = 1.0,
        thetaSigma         = 9.0,
        phiSigma           = 1.5,
        thetaTau           = 2.05,
        phiTau             = phi_tau,
        alphas             = alphas,
        randomWalkVariance = 9.0
    )

    # ---------- posterior summaries (after burn-in) ----------
    keep         = slice(burn_in, None)
    betaIN_hat   = betaIN_chain[keep].mean()
    betaOUT_hat  = betaOUT_chain[keep].mean()
    tauSq_hat    = tauSq_chain[keep].mean()
    sigmaSq_hat  = sigmaSq_chain[keep].mean()
    r_hat        = R_chain[keep].mean(axis=0)           # actor-wise mean
    X_hat        = X_chain[keep].mean(axis=0)           # latent positions

    # ---------- print comparison ----------
    print("\n=== PARAMETER RECOVERY CHECK ===")
    print(f"true  betaIN  = 1.0   |  posterior mean = {betaIN_hat:6.3f}")
    print(f"true  betaOUT = 2.0   |  posterior mean = {betaOUT_hat:6.3f}")
    print(f"true  SigmaSq = {SigmaSq:6.4f} |  posterior mean = {sigmaSq_hat:6.4f}")
    print(f"posterior mean of tauSq (true value varied each i) = {tauSq_hat:6.3f}")
    print("first five posterior means of r:", r_hat[:5])

    # ---------- (optional) save full MCMC output ----------
    np.savez_compressed("sim_run.npz",
                        X_true=LargeX, Y=Y,
                        X_chain=X_chain, R_chain=R_chain,
                        betaIN_chain=betaIN_chain, betaOUT_chain=betaOUT_chain,
                        tauSq_chain=tauSq_chain, sigmaSq_chain=sigmaSq_chain)
    print("Saved full chains to sim_run.npz")


'''






# Entry point check: only run main() if this script is executed directly
if __name__ == "__main__":  # Check if script is main program
    main()  # Call the main function to execute