import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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
    
    def BuildGlobalTracePlots(self, thinning = 1, showTruth = False):
        '''
        Builds a trace plot and outputs to self.outPath for each global variable (betaIN, betaOUT, tauSq, sigmaSq)
        Every {thinning} steps in the Markov chain will be used
        If showTruth, the true value passed in the constructor will be plotted as well.
        '''
        stepIndices = range(0, self.ns, thinning)
        
        for (data, name, trueValue) in [(self.betaINChain, "Beta_IN", self.trueBetaIN), 
                                        (self.betaOUTChain, "Beta_OUT", self.trueBetaOUT),
                                        (self.tauSqChain, "Tau Squared", self.trueTauSq), 
                                        (self.sigmaSqChain, "Sigma Squared", self.trueSigmaSq)]:
            plt.figure(figsize=(10, 6))
            plt.plot(data[stepIndices])
            plt.title(f"Trace Plot for {name} (n={self.n}, T={self.T}, p={self.p})")
            if thinning != 1:
                plt.xlabel(f"Iteration")
            else:
                plt.xlabel(f"Iteration (thinning every {thinning} steps)")
            if showTruth:
                plt.axhline(y=trueValue, color='red', linestyle='--', label='True Value')
            plt.savefig(os.path.join(self.outPath, f"TracePlot - {name}.png"))
            plt.close()

    def BuildGlobalHistograms(self, showTruth = False, binMethod = "sturges", burnIn = 0):
        '''
        For all global variables, build and output (to self.outPath) histograms for the distribution using the 
            specified bin-determining method (and ignoring anything in the burn-in period)
        If showTruth == True, plot the true values as well
        binMethod must be one of the methods that plt.hist() can take
        '''
        for (data, name, trueValue) in [(self.betaINChain, "Beta_IN", self.trueBetaIN), 
                                         (self.betaOUTChain, "Beta_OUT", self.trueBetaOUT),
                                         (self.tauSqChain, "Tau Squared", self.trueTauSq), 
                                         (self.sigmaSqChain, "Sigma Squared", self.trueSigmaSq)]:
            plt.figure(figsize=(10, 6))
            plt.title(f"Histogram for {name}")
            cutData = data[burnIn:]
            plt.hist(cutData, bins=binMethod)
            if showTruth:
                plt.axvline(x=trueValue, color='red', linestyle='--', linewidth=2, label='True Value')
            plt.savefig(os.path.join(self.outPath, f"Histogram - {name}.png"))
            plt.close()
    
    def BuildRadiusTracePlot(self, i, showTruth = False, thinning = 1):
        '''
        For the index-i actor, plot its radius trace plot over each MCMC iteration, every {thinning} steps.
        If showTruth, plot the true values as well
        Output results to self.outPath
        '''
        stepIndices = range(1, self.ns, thinning)
        plt.figure(figsize=(10, 6))
        plt.title(f"Trace Plot for Index {i} Actor's Radius")
        plt.plot(self.RChain[:, i])
        if thinning != 1:
            plt.xlabel(f"Iteration")
        else:
            plt.xlabel(f"Iteration (thinning every {thinning} steps)")
        if showTruth:
            plt.axhline(y=self.trueR[i], color='red', linestyle='--', linewidth=2, label='True Value')
        plt.savefig(os.path.join(self.outPath, f"Radius Trace Plot - Actor Index {i}.png"))
        plt.close()
    
    def BuildPositionTracePlot(self, i, t, showTruth = False, thinning = 1):
        '''
        For the index-i actor, plot its position over each MCMC iteration, every {thinning} steps.
        If showTruth, plot the true value as well
        If self.p = 1, this will be a 2D plot; if self.p = 2, this will be a 3D plot
        '''
        if self.p > 2:
            raise NotImplementedError # we cannot plot a p > 2 because there is an additional iteration dimension
        
        stepIndices = range(0, self.ns, thinning)
        # Cut down the XChain to only the specific actor and time, and only the suggested step indices.
        data = self.XChain[stepIndices, t, i, :] 

        if self.p == 2:
            # We are graphing this case in 3D (2 dimensions for position, 1 for time)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Select the specific aspects of data
            latentDimX = data[:, 0]
            latentDimY = data[:, 1]
            ax.plot(latentDimX, latentDimY, stepIndices, color='blue', linewidth=2, label="Chain Values")
            if showTruth:
                trueDimX = self.trueX[t, i, 0]*np.ones(self.ns)
                trueDimY = self.trueX[t, i, 1]*np.ones(self.ns)
                ax.plot(trueDimX, trueDimY, np.arange(self.ns), label="True Value", color="red", linewidth=2)
            ax.set_xlabel("Latent X Position")
            ax.set_ylabel("Latent Y Position")
            ax.set_zlabel("Gibbs Iteration")
            ax.set_title(f"Position Trace Plot - Actor Index {i}")
            plt.savefig(os.path.join(self.outPath, f"Position Trace Plot - Actor Index {i}.png"))
            plt.close()
        
        elif self.p == 1:
            # We are graphing this case in 2D (1 dimension for position, 1 for time)
            plt.figure(figsize=(8, 6))
            plt.plot(data[:, 0], color='blue', linewidth=2, label="Chain Values")
            if showTruth:
                trueValues = self.trueX[t, i, 0]*np.ones(self.ns)
                ax.plot(trueValues[stepIndices], label="True Value", color="red", linewidth=2)
            plt.xlabel("Gibbs Iteration")
            plt.ylabel("Latent Position")
            plt.title(f"Position Trace Plot - Actor Index {i}")
            plt.savefig(os.path.join(self.outPath, f"Position Trace Plot - Actor Index {i}.png"))
            plt.close()
    
    def BuildPositionDynamicPlot(self, i, showTruth = False, burnIn = 0):
        '''
        For the index-i actor, plot the estimate of the actor's latent position over time indices 0 ... T
            (the average of the MCMC estimates after the specified burnIn)
            (if self.p == 2, will be a 3D plot; if self.p == 1, will be a 2D plot)
        If showTruth == True, plot the true values as well
        Output result to self.outPath
        '''
        averagePositions = np.mean(self.XChain[burnIn:, :, i, :], axis=0) # Only the i-th actor after burn-in
        # Result of averagePositions has shape [T, p] (as we would want)
        
        # We don't have a way to plot any case where p > 2
        if self.p > 2:
            raise NotImplementedError

        # Plotting in the 3D case (p == 2)
        elif self.p == 2:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            dim1Estimate = averagePositions[:, 0] # all time periods, first coordinate
            dim2Estimate = averagePositions[:, 1] # all time periods, second coordinate
            timeData = np.arange(self.T)      # number of time periods
            ax.plot(dim1Estimate, dim2Estimate, timeData, color="blue", label="Chain Averages", linewidth=2)
            if showTruth:
                # Need to plot the true values (from self.trueX, which has shape (T, n, p))
                trueDim1Data = self.trueX[:, i, 0]
                trueDim2Data = self.trueX[:, i, 1]
                ax.plot(trueDim1Data, trueDim2Data, timeData, color="red", label="True Positions", linewidth=2)
            ax.set_xlabel("Latent X Position")
            ax.set_ylabel("Latent Y Position")
            ax.set_zlabel("Time")
            ax.set_title(f"Average Position of Index {i} Actor Over Time")
            plt.savefig(os.path.join(self.outPath, f"Dynamic Position Plot - Actor Index {i}.png"))
            plt.close()
        
        elif self.p == 1:
            # We are graphing this case in 2D (1 dimension for position, 1 for time)
            plt.figure(figsize=(8, 6))
            # Take the mean along the ns (will result in T-length)
            dim1Estimate = np.mean(self.XChain[burnIn:, :, i, 0], axis=0)
            plt.plot(dim1Estimate, color="blue", label="Chain Averages", linewidth=2)
            if showTruth:
                # Need to plot the true values (from self.trueX, which has shape (T, n, p))
                trueData = self.trueX[:, i, 0]
                ax.plot(trueData, color="red", label="True Positions", linewidth=2)
            plt.xlabel("Time")
            plt.ylabel("Latent Position")
            plt.title(f"Average Position of Index {i} Actor Over Time")
            plt.savefig(os.path.join(self.outPath, f"Dynamic Position Plot - Actor Index {i}.png"))
            plt.close()
    
    def BuildLogLikelihoodPlot(self, conditionals, thinning = 1):
        '''
        (require an instance of the conditionals class to determine what log-likelihood to use)
        Determine the indices to plot the log-likelihood at (as determined by thinning argument)
            (it's costly to compute log-likelihood; this should speed it up)
        Calculate the log-likelihood at each index required using the conditionals instance
        Plot the log-likelihoods against the iteration
        Output result to self.outPath
        '''
        # Pick every {thinning}-th index to calculate log-likelihood at and plot
        plotIndices = range(0, self.ns, thinning)

        # Create an empty vector of log-likelihoods and fill it in
        logLikelihoods = np.zeros(len(plotIndices))
        for i, plotIndex in zip(range(len(plotIndices)), plotIndices):
            logLikelihoods[i]  = conditionals.LogLikelihood(self.Y, 
                                                            self.XChain[plotIndex],
                                                            self.RChain[plotIndex],
                                                            self.betaINChain[plotIndex],
                                                            self.betaOUTChain[plotIndex],
                                                            self.tauSqChain[plotIndex],
                                                            self.sigmaSqChain[plotIndex])
        
        # Plot the values
        plt.figure(figsize=(8, 6))
        plt.plot(logLikelihoods, color="blue", linewidth=2)
        plt.title("Log-Likelihood over Gibbs Iterations")
        plt.xlabel("Gibbs Iteration")
        plt.ylabel("Log-Likelihood")
        plt.savefig(os.path.join(os.getcwd(), f"Log-Likelihood Plot.png"))
        plt.close()