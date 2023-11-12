import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neural_network import MLPRegressor

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
eta = 0.01
lam = 1e-05
rho = 0.9
rho2 = 0.99
batches = 32
nodes_in_layer = int(X.shape[1] + 1 / 2)
layers_to_try = 4

scheduler = Adam(eta, rho, rho2)
hidden_func = LRELU

n_layers_scores, layer = optimize_n_hidden_layers(X_train, z_train, folds, scheduler, batches, epochs, lam, nodes_in_layer, layers_to_try, hidden_func)


for i in range(len(n_layers_scores)):
    lab = f"Hidden layer: {layer[i]}"
    plt.plot(n_layers_scores[i]["val_errors"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

#getting the length for next optimization
min_error = float('inf')  
best_layer_length = None

for i in range(len(n_layers_scores)):
    current_error = min(n_layers_scores[i]["val_errors"])
    
    if current_error < min_error:
        min_error = current_error
        best_layer_length = len(layer[i])

#Try powers of two in nodes
nodes_to_try = np.power(2, np.arange(7))
#add the number of nodes we initially tested with
nodes_to_try = np.insert(nodes_to_try, 6, nodes_in_layer)

n_nodes_scores, node_layer = optimize_n_nodes(X_train, z_train, folds, scheduler, batches, epochs, lam, best_layer_length, nodes_to_try, hidden_func)

for i in range(len(n_nodes_scores)):
    lab = f"Hidden layer: {node_layer[i]}"
    plt.plot(n_nodes_scores[i]["val_errors"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

"""
Hidden layer for LRELU: (64, 64)
Hidden layer for RELU: (45, 45, 45, 45)
Hidden layer for sigmoid: (64, 64, 64, 64)
"""