import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neural_network import MLPRegressor

from schedulers import *
from FFNN import *
from utils import *

np.random.seed(1984)

#Step size between 1, 0. Number of points for 0.05 = 20.
step = 0.05
#Beta values 
betas = 10
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use Skranke
use_franke = True
#Max polynomial degree
maxDegree = 10
#Number of epochs
epochs = 200
folds = 5
batches = 20
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, step, maxDegree)
eta = 0.001
lam = 0.0001
rho = 0.9
rho2 = 0.99

scheduler = Adam(eta, rho, rho2)
"""
n_layers_scores, layer = optimize_n_hidden_layers(X, z, folds, scheduler, batches, epochs, lam, 60, 6)


for i in range(len(n_layers_scores)):
    lab = f"Hidden layer: {layer[i]}"
    plt.plot(n_layers_scores[i]["val_errors"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
"""
#Try powers of two in nodes
nodes_to_try = np.power(2, np.arange(9))

#Try linspace from 30 to 65
#nodes_to_try = np.array([30, 35, 40, 45, 50, 55, 65])
n_nodes_scores, node_layer = optimize_n_nodes(X, z, folds, scheduler, batches, epochs, lam, 2, nodes_to_try)

for i in range(len(n_nodes_scores)):
    lab = f"Hidden layer: {node_layer[i]}"
    plt.plot(n_nodes_scores[i]["val_errors"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()