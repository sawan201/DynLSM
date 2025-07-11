import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf
import os
import conditionalposteriors as cp

class BinaryDiagnostics:
    def __init__(self, simResultsPath, outPath, modelType = "binary", truthIncluded = True):
        '''
        Creates a BinaryDiagnostics object.
        simResultsPath = path to the exact simResults .npz file (not the folder it is located in)
        outPath = path to the folder you would like the plots dropped into
        modelType = "binary" or "poisson" depending on likelihood function needed
        truthIncluded = should be true if there are values in the .npz file containing true values for parameters
        '''
        # simResultsPath is the path leading to an .npz file where matrices/tensors with the following names are:
        # "Y", "X_Chain", "R_Chain", "betaIN_Chain", "betaOUT_Chain", "tauSqChain", "sigmaSqChain"
        self.simResults = np.load(simResultsPath)
        # outPath = where to store visualizations
        self.outPath = outPath
        # Determine which type of conditionals to use based on the model type specified
        # We only need the conditionals for the likelihood function, not for any of their attributes
        if modelType == "binary":
            self.conditionals = cp.BinaryConditionals(0, 0, 0, 0, 0, 0, 0, 0)
        else:
            print("Model Type incorrectly specified. Use 'binary' for binary conditionals.")
        # truth = if a simulation study, this is a dictionary containing the true values
        if truthIncluded:
            self.trueX = self.simResults["trueX"]
            self.trueR = self.simResults["trueR"]
            self.trueBetaIN = float(self.simResults["trueBetaIN"])
            self.trueBetaOUT = float(self.simResults["trueBetaOUT"])
            self.trueSigmaSq = float(self.simResults["trueSigmaSq"])
            self.trueTauSq = float(self.simResults["trueTauSq"])
        
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
                plt.plot(trueValues[stepIndices], label="True Value", color="red", linewidth=2)
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
                plt.plot(trueData, color="red", label="True Positions", linewidth=2)
            plt.xlabel("Time")
            plt.ylabel("Latent Position")
            plt.title(f"Average Position of Index {i} Actor Over Time")
            plt.savefig(os.path.join(self.outPath, f"Dynamic Position Plot - Actor Index {i}.png"))
            plt.close()
    
    def BuildLogLikelihoodPlot(self, thinning = 1, conditionals = None):
        '''
        (require an instance of the conditionals class to determine what log-likelihood to use)
        Determine the indices to plot the log-likelihood at (as determined by thinning argument)
            (it's costly to compute log-likelihood; this should speed it up)
        Calculate the log-likelihood at each index required using the conditionals instance
        Plot the log-likelihoods against the iteration
        Output result to self.outPath
        '''
        if conditionals is None:
            conditionals = self.conditionals
        
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

    def BuildParameterEstimates(self, showTruth = False, burnIn = 0):
        '''
        For each parameter, determine the average value for the Markov chain after burnIn number of steps
        (a scalar value for global parameters, a n-length vector for R, and a T x n x p tensor for positions)
        If showTruth == True, include the true values and the difference between estimated and true
        Output these results to a .txt file (or similar, maybe .npz?)
        '''
        # Estimate the parameters
        betaINEstimate = np.mean(self.betaINChain[burnIn:])
        betaOUTEstimate = np.mean(self.betaOUTChain[burnIn:])
        tauSqEstimate = np.mean(self.tauSqChain[burnIn:])
        sigmaSqEstimate = np.mean(self.sigmaSqChain[burnIn:])
        radiiEstimate = np.mean(self.RChain[burnIn:, :], axis=0)
        positionEstimate = np.mean(self.XChain[burnIn:, :, :, :], axis=0)

        if showTruth:
            # Save the globals and radii to a .txt file
            outputString = f'''
            Global Parameters
            betaIN Estimate:    {betaINEstimate}    (true: {self.trueBetaIN}, difference {betaINEstimate - self.trueBetaIN})
            betaOUT Estimate:   {betaOUTEstimate}   (true: {self.trueBetaOUT}, difference {betaOUTEstimate - self.trueBetaOUT})
            tauSq Estimate:     {tauSqEstimate}     (true: {self.trueTauSq}, difference {tauSqEstimate - self.trueTauSq})
            sigmaSq Estimate:   {sigmaSqEstimate}   (true: {self.trueSigmaSq}, difference {sigmaSqEstimate - self.trueSigmaSq})
            
            Radii Parameters:
            '''
            for i in range(self.n):
                outputString += f"\nRadius Estimate for Index {i} Actor: {radiiEstimate[i]}     (true: {self.trueR[i]}, difference {radiiEstimate[i] - self.trueR[i]})"

            # Save the estimates themselves
            np.savez(f"Estimates_ns{self.ns}_T{self.T}_n{self.n}_p{self.p}_burn{burnIn}.npz",
                    betaINEstimate = betaINEstimate,
                    betaOUTEstimate = betaOUTEstimate,
                    tauSqEstimate = tauSqEstimate,
                    sigmaSqEstimate = sigmaSqEstimate,
                    radiiEstimate = radiiEstimate,
                    positionEstimate = positionEstimate,
                    trueBetaIN = self.trueBetaIN,
                    trueBetaOUT = self.trueBetaOUT,
                    trueSigmaSq = self.trueSigmaSq,
                    trueTauSq = self.trueTauSq,
                    trueRadii = self.trueR,
                    truePositions = self.trueX)

        else:
            # Save the globals and radii to a .txt file
            outputString = f'''
            Global Parameters
            betaIN Estimate:    {betaINEstimate}
            betaOUT Estimate:   {betaOUTEstimate}
            tauSq Estimate:     {tauSqEstimate}
            sigmaSq Estimate:   {sigmaSqEstimate}
            
            Radii Parameters:
            '''
            for i in range(self.n):
                outputString += f"\nRadius Estimate for Index {i} Actor: {radiiEstimate[i]}"

            # Save to a .npz file
            np.savez(f"Estimates_ns{self.ns}_T{self.T}_n{self.n}_p{self.p}_burn{burnIn}_NoTruth.npz",
                    betaINEstimate = betaINEstimate,
                    betaOUTEstimate = betaOUTEstimate,
                    tauSqEstimate = tauSqEstimate,
                    sigmaSqEstimate = sigmaSqEstimate,
                    radiiEstimate = radiiEstimate,
                    positionEstimate = positionEstimate)
        
        with open(os.path.join(os.getcwd(), f"Estimates after {burnIn} Burn-In Values.txt"), "w+") as writeFile:
            writeFile.write(outputString)
    
    def BuildGlobalAutocorrelationPlots(self, burnIn = 0, maxLag = None):
        '''
        For each global parameter, build an autocorrelation plot where the maximumLag is as specified and the specified
        burnIn period is removed.
        '''
        if maxLag is None:
            maxLag = self.ns - burnIn - 1
        
        # Build dictionary of parameter values to use
        global_chains = {
            'betaIN':   self.betaINChain[burnIn:],
            'betaOUT':  self.betaOUTChain[burnIn:],
            'tauSq':    self.tauSqChain[burnIn:],
            'sigmaSq':  self.sigmaSqChain[burnIn:]
            }

        # Step 2: Create subplots: one for each parameter
        num_params = len(global_chains)
        fig, axs = plt.subplots(num_params, 1, figsize=(8, 2.5 * num_params))
        
        if num_params == 1:
            axs = [axs]  # ensure axs is always iterable

        # Step 3: Loop through each parameter and plot its autocorrelation
        for ax, (name, chain) in zip(axs, global_chains.items()):
            plot_acf(chain, ax=ax, lags=maxLag)
            ax.set_title(f'Autocorrelation for {name}')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')

        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "Autocorrelation Plots - Global.png"))
        plt.close()