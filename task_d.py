import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from schedulers import *
from FFNN import *
from utils import *

seed = np.random.seed(4231)

#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.0
#If True use Franke, if False use cancer data
use_franke = False
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5
x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)

hidden_layer = (64, 64, 64, 64)
eta = 0.0001
lam = 1e-05
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
scheduler = Adam(eta = eta, rho = rho, rho2 = rho2)
#best values for every scheduler
ffnn = FFNN((X.shape[1], *hidden_layer, 1), seed = seed, cost_func=CostLogReg, output_func=sigmoid, hidden_func=sigmoid)

scores = ffnn.cross_validation(X, z, folds, scheduler, batches, epochs, lam)

sns.heatmap(scores["confusion_matrix"], annot=True, fmt = ".3%",  cmap='Greens')
plt.title("Confusion matrix of breast cancer dataset after ffnn fitting")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.text(0.5, 0.2, "True Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(0.5, 1.2, "False Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(1.5, 0.2, "False Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(1.5, 1.2, "True Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.show()

