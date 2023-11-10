import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *

seed = np.random.seed(4231)

#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use Skranke
use_franke = True
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, 1 / datapoints, maxDegree)
ffnn = FFNN((X.shape[1], 1), seed = seed)
etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
scheduler_list = [
    "Adam",
    "Constant",
    "Momentum",
    "Adagrad",
    "AdagradMomentum",
    "RMS_prop",
]

#best values for every scheduler
best_etas = np.zeros(6)
best_lambdas = np.zeros(6)
i = 0

for s in scheduler_list:
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, z_train, etas, lambdas, s, batches = batches, epochs = epochs, momentum=momentum, rho=rho, rho2=rho2, folds = folds)
    print(f"\n Best eta for {s}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    plt.title(f"{s}, average validation error over {folds} folds")
    plt.show()
    best_etas[i] = best_eta
    best_lambdas[i] = best_lambda
    i += 1


"""
|Scheduler      |eta |lambda      |mse   |
|Constant       |0.1 |1e-5/0.00001|0.0146|
|Momentum       |0.1 |1e-5        |0.0105|
|Adagrad        |0.1 |0.01        |0.0264|
|AdagradMomentum|0.1 |0.001       |0.0165|
|RMS prop       |0.01|1e'5        |0.0174|
|Adam           |0.01|0.001       |0.0167|
"""


