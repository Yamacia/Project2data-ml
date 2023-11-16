import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *
"""
This script optimizes hyperparameters for every scheduler or the classification case
"""
seed = np.random.seed(4231)

#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use cancer data
use_franke = False
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5

x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)
ffnn = FFNN((X.shape[1], 1), seed = seed, cost_func=CostCrossEntropy, output_func=sigmoid)
etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
scheduler_list = [
    "Adam",
    # "Constant",
    # "Momentum",
    # "Adagrad",
    # "AdagradMomentum",
    # "RMS_prop",
    # "RMS_propMomentum",
    # "AdamMomentum",
]

#best values for every scheduler
best_etas = np.zeros(8)
best_lambdas = np.zeros(8)
i = 0

for s in scheduler_list:
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, z_train, etas, lambdas, s, batches = batches, epochs = epochs, momentum=momentum, rho=rho, rho2=rho2, folds = folds)
    print(f"\n Best eta for {s}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    type = "accuracy" if ffnn.classification else "score"
    plt.title(f"{s}, average validation {type} over {folds} folds")
    results_path = f'{s}_results_cancer.png'
    plt.savefig(results_path)
    plt.show()
    plt.close()
    best_etas[i] = best_eta
    best_lambdas[i] = best_lambda
    i += 1