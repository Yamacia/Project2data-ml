import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *


"""
NEEDS TO BE OPTIMIZED FOR EVERY ACTIVATION FUNCTION
"""
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
x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)
batches = 32
rho = 0.9
rho2 = 0.99
momentum = 0.5


hidden_funcs = [
    sigmoid, 
    RELU, 
    LRELU]

hidden_layers = [
    (64, 64, 64, 64),
    (45, 45, 45, 45),
    (64, 64)
]

lambdas = [
    1e-05,
    0.0001,
    1e-05
]

etas = [
    0.0001,
    0.01,
    0.01
]

func_string = [
    "sigmoid", 
    "RELU", 
    "LRELU"]
scores = run_funcs(X, z, folds, batches, epochs, etas, lambdas, hidden_layers, hidden_funcs)

colors = ['yellow', 'purple', 'green']

for i in range(len(scores)):
    plt.plot(scores[i]["train_errors"], label=f"Train set {func_string[i]}", linestyle='-', color=colors[i])
    plt.plot(scores[i]["val_errors"], label=f"Val set {func_string[i]}", linestyle='--', color=colors[i])

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE as a function of Epochs for our FFNN, using different activation functions")
plt.show()