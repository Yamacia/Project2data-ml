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

etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32


hidden_funcs = [sigmoid, RELU, LRELU]

scheduler = "Adam"

for func in hidden_funcs:
    ffnn = FFNN(dimensions=(X.shape[1], 1), hidden_func=func, seed=4231, output_func= lambda x: x)
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, z_train, etas, lambdas, scheduler, batches = batches, epochs = epochs, momentum=momentum, rho=rho, rho2=rho2, folds = folds)
    print(f"\n Best eta for {scheduler} with {func}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    plt.title(f"{scheduler}, average validation error over {folds} folds using activation function: {func}")
    plt.show()