import simulation

def main():
    # Initialize the simulation parameters
    T = 3  # Number of time points
    n = 3   # Number of actors
    p = 1   # Latent space dimensions
    SigmaSq = 1.0  # Variance for the latent positions
    TauSq = 1.0  # Variance for the tau prior
    ThetaTau = 2.05  # Shape parameter for tau prior
    ThetaSigma = 9.0  # Shape parameter for sigma prior
    PhiSigma = 1.5  # Scale parameter for sigma prior
    NuIn = 0.0  # Input effect parameter for the model
    XiIn = 1.0  # Input effect parameter for the model
    NuOut = 0.0  # Output effect parameter for the model
    XiOut = 1.0  # Output effect parameter for the model
    EtaIn = 1.0  # Input effect parameter for the model
    EtaOut = 1.0  # Output effect parameter for the model
    BetaIn = 0.8  # Input effect parameter for the model
    BetaOut = 1.3  # Output effect parameter for the model
    RandomWalkVariance = 100.0  # Variance for the random walk 
    DirichletFactor = 1200  # Factor for the Dirichlet prior 
    model_type = "binary"  # Type of model (e.g., "binary")
    InitType = "base"  # Initialization type for the model
    NumberOfSamples = 1000  # Total MCMC sweeps
    BurnIn = 100  # First draws to discard
    simName = "test"


    sim = simulation.Simulation(T, n, p, SigmaSq, TauSq, ThetaTau, ThetaSigma, PhiSigma, NuIn, XiIn, NuOut, XiOut, EtaIn, EtaOut, BetaIn, BetaOut, RandomWalkVariance, DirichletFactor, model_type, InitType, NumberOfSamples, BurnIn, simName = simName)
    
    sim.run()  # Run the simulation

if __name__ == "__main__":
    main()  # Execute the main function when the script is run