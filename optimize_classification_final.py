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

starting_layer = [64, 64, 64, 64]
try_nodes = [50, 64, 78, 92]
adam = Adam(eta, rho, rho2)
best_accuracy = 0
for i in range(len(starting_layer)):
    curr_scores = np.zeros(len(try_nodes))
    
    for j, nodes in enumerate(try_nodes):
        starting_layer[i] = nodes
        hidden_layer = tuple(starting_layer)
        
        ffnn = FFNN((X.shape[1], *hidden_layer, 1), seed=seed, cost_func=CostLogReg, output_func=sigmoid, hidden_func=sigmoid)
        
        scores = ffnn.cross_validation(X, z, folds, adam, batches, epochs, lam)
        acc = np.max(scores["val_accs"])
        
        curr_scores[j] = acc
        print(f"Current hidden layer {hidden_layer}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_nodes = nodes
            
    starting_layer[i] = best_nodes



print("Best Accuracy:", best_accuracy)
print("Best Hidden Layer Configuration:", starting_layer)