import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *
from sklearn.neural_network import MLPRegressor

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
epochs = 200
#Number of folds for cross validation
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, 1 / datapoints, maxDegree)
rho = 0.9
rho2 = 0.99
momentum = 0.5
batches = 32
#hidden layer configuration gotten from optimize_nodes.py and optimize_hidden_func_adam.py
adam = Adam(eta = 0.0001 , rho = rho, rho2 = rho2)
hidden_layer_sizes = (64, 64, 64, 64)
output_layer_size = 1

mlp_regressor = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='logistic',
    solver='adam',
    max_iter=1000,
    alpha=0.001,
    random_state=4231,
    batch_size = X_train.shape[0] // 30,
    early_stopping=False,
    n_iter_no_change=epochs + 1
)

mlp_regressor.fit(X_train, np.ravel(z_train))
mlp_scores = mlp_regressor.loss_curve_


ffnn = FFNN((X.shape[1] , *hidden_layer_sizes, output_layer_size), hidden_func=sigmoid, seed=1984, output_func= lambda x: x)

scores_adam = ffnn.cross_validation(X, z.reshape(-1, 1), folds, adam, batches, epochs, lam = 1e-05)




plt.plot(scores_adam["train_errors"], label="Train set error FFNN Adam")
plt.plot(scores_adam["val_errors"], label="Val set error FFNN Adam")
plt.plot(range(200), mlp_scores[:200], label="scikit error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE as a function of Epochs for our FFNN and Scikit MLP regressor using Adam")
plt.show()

