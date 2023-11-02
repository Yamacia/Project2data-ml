import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
epochs = 1000
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, step, maxDegree)

ffnn = FFNN((X.shape[1], 1), seed = 1984)

etas = np.logspace(-5, -1, 5)
lambdas = np.logspace(-5, -1, 5)
rho = 0.9
rho2 = 0.999
momentum = 0.03
batches = 1

scheduler_list = [
    "Constant",
    "Momentum",
    "Adagrad",
    "AdagradMomentum",
    "RMS_prop",
    "Adam"
]

#best values for every scheduler
best_etas = np.zeros(6)
best_lambdas = np.zeros(6)
best_batches = np.zeros(6)
i = 0

for s in scheduler_list:
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, z_train, etas, lambdas, s, batches, epochs, momentum=momentum, rho=rho, rho2=rho2, folds=5)
    print(f"\n Best eta for {s}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True)
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    plt.title(f"{s}, average validation score over {folds} folds")
    plt.show()
    best_etas[i] = best_eta
    best_lambdas[i] = best_lambda
    i += 1





