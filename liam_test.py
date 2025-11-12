import numpy as np
import simulation
import os 

simName = "LiamFullSubsequenceTest11_10_25"
nsamples = 30000
burnIn = 5000

# Initialize the simulation parameters
T = 4  # Number of time points
n = 10   # Number of actors
p = 2   # Latent space dimensions
SigmaSq = 0.0004  # 1/5n^2
TauSq = 0.0004  # Variance for the tau prior
ThetaTau = 20.0  # Shape parameter for tau prior
ThetaSigma = 20.0  # Shape parameter for sigma prior
PhiSigma = 0.038  # Scale parameter for sigma prior
NuIn = 1.0  # Input effect parameter for the model
XiIn = 0.2  # Input effect parameter for the model
NuOut = 2.0  # Output effect parameter for the model
XiOut = 0.2  # Output effect parameter for the model
BetaIn = 1  # Input effect parameter for the model
BetaOut = 2  # Output effect parameter for the model
DirichletFactor = 40000  # Factor for the Dirichlet prior 
model_type = "case_control_binary"  # Type of model (e.g., "binary")
subsequence_length = 9 # length of the subsequence to analyze for case-control    

print("Started")

sim = simulation.Simulation(T, n, p, SigmaSq, TauSq, ThetaTau, ThetaSigma, PhiSigma, 
                            NuIn, XiIn, NuOut, XiOut, BetaIn, BetaOut, model_type)

sim.run(simName = simName, numberOfSamples = nsamples, burnIn = burnIn, initType = "truth", 
        betaRandomWalkVariance = 0.01, 
        positionRandomWalkVariance=0.01,
        dirichletFactor = DirichletFactor,
        randomSeed=42,
        fixX = False, fixR = False, fixBetaIN = False, fixBetaOUT = False, fixSigmaSq = False, fixTauSq = False,
        subsequence_length = subsequence_length)


fname     = f"sim_run_{model_type}{simName}_ns{nsamples}_T{T}_n{n}_p{p}.npz"
npz_path  = os.path.join(os.getcwd(), fname)

with np.load(npz_path) as data:
    Y = data["Y"]          # (T, n, n) tensor of 0/1

density = Y.mean()         # proportion of ones
print(f"{density*100:.4f}% of Y entries are 1")
