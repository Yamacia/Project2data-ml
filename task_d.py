import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import warnings
from sklearn.exceptions import ConvergenceWarning

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

# scores = ffnn.cross_validation(X, z, folds, scheduler, batches, epochs, lam)

# sns.heatmap(scores["confusion_matrix"], annot=True, fmt = ".3%",  cmap='Greens')
# plt.title("Confusion matrix of breast cancer dataset after ffnn fitting")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.text(0.5, 0.2, "True Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
# plt.text(0.5, 1.2, "False Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
# plt.text(1.5, 0.2, "False Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
# plt.text(1.5, 1.2, "True Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
# plt.show()

# Scikit-Learn comparison
warnings.filterwarnings("ignore", category=ConvergenceWarning)
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

kf = KFold(n_splits = folds)

dnn = MLPClassifier(solver = "adam", hidden_layer_sizes=(78,78,78,78), activation='logistic',
                    alpha=lam, learning_rate_init=eta, max_iter=epochs, batch_size= batches, momentum=momentum)

confusion = 0
confusion_matrix = np.zeros((2, 2))

for train_index, test_index in kf.split(X_scaled):
    dnn.fit(X_scaled[train_index],np.ravel(z[train_index]))
    print()
    print("Accuracy Score (training): ", dnn.score(X_scaled[train_index],z[train_index]))

    zpredict = dnn.predict(X_scaled[test_index])
    confusion = calc_confusion(np.ravel(z[test_index]), zpredict)
    confusion_matrix += confusion / folds
    print("Accuracy Score (test): ", accuracy_score(z[test_index], zpredict))

sns.heatmap(confusion_matrix, annot=True, fmt = ".3%",  cmap='Greens')
plt.title("Confusion matrix of breast cancer dataset using K-folded Scikit-Learn")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.text(0.5, 0.2, "True Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(0.5, 1.2, "False Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(1.5, 0.2, "False Negative", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.text(1.5, 1.2, "True Positive", horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
plt.show()