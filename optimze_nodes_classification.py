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
use_franke = False
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)
eta = 0.0001
lam = 1e-05
rho = 0.9
rho2 = 0.99
batches = 32

nodes_in_layer = int(X.shape[1] + 1 / 2)
layers_to_try = 4

adam = Adam(eta, rho, rho2)
hidden_func = RELU

n_layers_scores, layer = optimize_n_hidden_layers(X, z, folds, adam, batches, epochs, lam, nodes_in_layer, layers_to_try, hidden_func = sigmoid, cost_func = CostLogReg,  output_func=sigmoid)

for i in range(len(n_layers_scores)):
    lab = f"Hidden layer: {layer[i]}"
    plt.plot(n_layers_scores[i]["val_accs"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#getting the length for next optimization
best_accuracy = 0
best_layer_length = None

for i in range(len(n_layers_scores)):
    current_accuracy = max(n_layers_scores[i]["val_accs"])
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_layer_length = len(layer[i])
#Try powers of two in nodes
nodes_to_try = np.power(2, np.arange(7))
#add the number of nodes we initially tested with
nodes_to_try = np.insert(nodes_to_try, 6, nodes_in_layer)

n_nodes_scores, node_layer = optimize_n_nodes(X_train, z_train, folds, adam, batches, epochs, lam, 4, nodes_to_try, hidden_func = sigmoid, cost_func = CostLogReg,  output_func=sigmoid)

for i in range(len(n_nodes_scores)):
    lab = f"Hidden layer: {node_layer[i]}"
    plt.plot(n_nodes_scores[i]["val_accs"], label = lab)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()