import numpy as np
import matplotlib.pyplot as plt
simRun = np.load("sim_run.npz")
betaIN = simRun["betaIN_chain"]

plt.hist(betaIN, width=0.1)
plt.show()