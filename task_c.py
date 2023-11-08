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
maxDegree = 10
#Number of epochs
epochs = 1000
#Number of folds for cross validation
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, 1 / datapoints, maxDegree)
batches = 32
hidden_layer = (64, 64, 64, 64, 64)
lam = 0.001
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
eta = 0.0001 
adam = Adam(eta=eta, rho=rho,rho2=rho2)

hidden_funcs = [sigmoid, RELU, LRELU]

scores = optimize_func(X, z, folds, adam, batches, epochs, lam, hidden_layer, hidden_funcs)

for i in range(len(scores)):
    plt.plot(scores[i]["train_errors"], label=f"Train set {hidden_funcs[i]}")
    plt.plot(scores[i]["val_errors"], label=f"Val set {hidden_funcs[i]}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE as a function of Epochs for our FFNN and Scikit MLP regressor using Adam")
plt.show()
