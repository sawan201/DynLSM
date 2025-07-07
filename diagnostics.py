import numpy as np
import matplotlib.pyplot as plt

class BinaryDiagnostics:
    def __init__(self, simResultsPath, outPath, conditionals, truth = None):
        # simResultsPath is the path leading to an .npz file where matrices/tensors with the following names are:
        # "Y", "X_Chain", "R_Chain", "betaIN_Chain", "betaOUT_Chain", "tauSqChain", "sigmaSqChain"
        self.simResults = np.load(simResultsPath)
        # outPath = where to store visualizations
        self.outPath = outPath
        # conditionals = whatever instance of the conditionals class was used when the model was running
        self.conditionals = conditionals
        # truth = if a simulation study, this is a dictionary containing the true values
        if truth:
            self.trueX = truth["trueX"]
            self.trueR = truth["trueR"]
            self.trueBetaIN = truth["trueBetaIN"]
            self.trueBetaOUT = truth["trueBetaOUT"]
            self.trueSigmaSq = truth["trueSigmaSq"]
            self.trueTauSq = truth["trueTauSq"]
        
        # Read in from .npz file
        self.Y = self.simResults["Y"]
        self.XChain = self.simResults["X_Chain"]
        self.RChain = self.simResults["R_Chain"]
        self.betaINChain = self.simResults["betaIN_Chain"]
        self.betaOUTChain = self.simResults["betaOUT_Chain"]
        self.tauSqChain = self.simResults["tauSqChain"]
        self.sigmaSqChain = self.simResults["sigmaSqChain"]
        self.ns = self.XChain.shape[0]  # number of steps in the Markov chain
        self.T = self.XChain.shape[1]   # number of timestamps
        self.n = self.XChain.shape[2]   # number of actors
        self.p = self.XChain.shape[3]   # dimension of latent space