import os
import diagnostics as d

# Need to fix the way that we show the truth. If the truth is available, we need to show it.

simRunName = "sim_run_case_control_binaryCaseControlTest_ns15000_T4_n10_p2"
npzName = simRunName + ".npz"
simResultsPath = os.path.join(os.getcwd(), "Simulation_Runs", npzName)

outPath = os.path.join(os.getcwd(), "Simulation_Diagnostics", "Diagnostics_" + simRunName)
os.mkdir(outPath)

modelType = "case_control_binary"
truthIncluded = True

# Diagnostic specific conditions
traceThinning = 100
likelihoodThinning = 100
burnIn = 5000

myDiagnostics = d.BinaryDiagnostics(simResultsPath, outPath, modelType, truthIncluded)
myDiagnostics.BuildAll(traceThinning, likelihoodThinning, burnIn)