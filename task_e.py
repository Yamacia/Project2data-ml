import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning

from schedulers import *
from FFNN import *
from utils import *

seed = np.random.seed(4231)

#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use cancer data
use_franke = False
#Max polynomial degree
maxDegree = 8
#Number of epochs
epochs = 500
#Number of folds for cross validation
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_dataset(use_franke, noise, 1 / datapoints, maxDegree)
ffnn = FFNN((X.shape[1], 1), seed = seed)
etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5, -1, 5)
lambdas = np.insert(lambdas, 0, 0)
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
scheduler_list = [
    "Constant",
]

#best values for every scheduler
best_etas = np.zeros(8)
best_lambdas = np.zeros(8)
i = 0

DNN_scikit = np.zeros((len(etas), len(lambdas)), dtype=object)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X_train_scaled = min_max_scaler.transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

for s in scheduler_list:
    #Set the cost_func to CostLogReg
    heatmap, best_eta, best_lambda = ffnn.optimze_params(X_train, z_train, etas, lambdas, s, batches = batches, epochs = epochs, momentum=momentum, rho=rho, rho2=rho2, folds = folds)
    print(f"\n Best eta for {s}: {best_eta}, Best lambda: {best_lambda}")
    ax = sns.heatmap(heatmap, xticklabels=lambdas, yticklabels=etas, annot=True, fmt = ".4f", cmap='viridis_r')
    plt.xlabel("lambda value")
    plt.ylabel("eta value")
    type = "accuracy" if ffnn.classification else "score"
    plt.title(f"{s}, average validation {type} over {folds} folds")
    results_path = f'{s}_results_franke.png'
    plt.savefig(results_path)
    plt.close()
    best_etas[i] = best_eta
    best_lambdas[i] = best_lambda
    i += 1

    # for i, eta in enumerate(etas):
    #     for j, lmbd in enumerate(lambdas):
    #         dnn = MLPRegressor(solver = "sgd", hidden_layer_sizes=64, activation='logistic',
    #                             alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
    #         dnn.fit(X_train_scaled, np.ravel(z_train))
    #         zpredict_train = dnn.predict(X_train_scaled)
    #         DNN_scikit[i][j] = dnn

    sns.set_theme()
    test_accuracy = np.zeros((len(etas), len(lambdas)))
    for i in range(len(etas)):
        for j in range(len(lambdas)):
            dnn = DNN_scikit[i][j]
            zpredict_test = dnn.predict(X_test_scaled)
            test_accuracy[i][j] = (1.0 / z_test.shape[0]) * np.sum((np.ravel(z_test) - zpredict_test) ** 2)

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, xticklabels=lambdas, yticklabels=etas, annot=True, ax=ax, fmt = ".4f", cmap="viridis_r")
    ax.set_title("Test Accuracy")
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\lambda$")
    plt.show()  


"""
|Scheduler      |eta |lambda      |mse   |
|Constant       |0.1 |1e-5/0.00001|0.0146|
|Momentum       |0.1 |1e-5        |0.0105|
|Adagrad        |0.1 |0.01        |0.0264|
|AdagradMomentum|0.1 |0.001       |0.0165|
|RMS prop       |0.01|1e'5        |0.0174|
|Adam           |0.01|0.001       |0.0167|
"""

