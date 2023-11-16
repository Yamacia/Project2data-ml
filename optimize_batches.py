import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *

"""
This script finds the optimal batches to use for Adam GD, with a signle hidden layer of the size int(X.shape[1] + 1)
"""
seed = np.random.seed(4231)

#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use Cancer dataset
use_franke = True
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5

x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)
ffnn = FFNN((X.shape[1], int(X.shape[1] + 1 / 2), 1), seed = seed)
lam = 0.001
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches_list = [8, 16, 32, 64, 128]
adam = Adam(eta = 0.01, rho=rho, rho2=rho2)
for batch in batches_list:
    scores = ffnn.cross_validation(X_train, z_train, folds, adam, batch, epochs, lam)
    lab = f"Batches: {batch}"
    plt.plot(scores["val_errors"], label = lab)

plt.title("MSE as a function of epochs")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()