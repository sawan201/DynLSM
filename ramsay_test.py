import numpy as np
import simulation
import os 

simName = "BadInit"
nsamples = 10000
burnIn = 2000

# Initialize the simulation parameters
T = 4  # Number of time points
n = 37   # Number of actors
p = 2   # Latent space dimensions
SigmaSq = 0.0004  # 1/5n^2
TauSq = 0.0004  # Variance for the tau prior
ThetaTau = 3.0  # Shape parameter for tau prior
ThetaSigma = 3.0  # Shape parameter for sigma prior
PhiSigma = 0.0012  # Scale parameter for sigma prior
NuIn = 1.0  # Input effect parameter for the model
XiIn = 100.0  # Input effect parameter for the model
NuOut = 2.0  # Output effect parameter for the model
XiOut = 100.0  # Output effect parameter for the model
BetaIn = 1  # Input effect parameter for the model
BetaOut = 2  # Output effect parameter for the model  # Factor for the Dirichlet prior 
model_type = "zero_inflated_poisson"  # Type of model (e.g., "binary", "poisson", "zero_inflated_poisson")
zeroInflationProb = 0.6  # Probability of structural zeros for ZIP model

sim = simulation.Simulation(T, n, p, SigmaSq, TauSq, ThetaTau, ThetaSigma, PhiSigma, NuIn, XiIn, NuOut, XiOut, BetaIn, BetaOut, model_type, zeroInflationProb=zeroInflationProb)

sim.run(simName = simName, randomSeed = 100, dirichletFactor = 15, positionRandomWalkVariance = 0.12, betaRandomWalkVariance = 0.08, numberOfSamples = nsamples, burnIn = burnIn, initType = "base", fixX = False, fixR = False, fixBetaIN = False, fixBetaOUT = False, fixSigmaSq = False, fixTauSq = False)


fname     = f"sim_run_{model_type}{simName}_ns{nsamples}_T{T}_n{n}_p{p}.npz"
npz_path  = os.path.join(os.getcwd(), fname)

with np.load(npz_path) as data:
    Y = data["Y"]          # (T, n, n) count tensor

zero_share = (Y == 0).mean()
mean_count = Y.mean()
print(f"{zero_share*100:.4f}% of Y entries are 0")
print(f"Mean Y entry: {mean_count:.6f}")
