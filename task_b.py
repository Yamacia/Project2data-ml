import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neural_network import MLPRegressor

from schedulers import *
from FFNN import *
from utils import *

np.random.seed(1984)

#Step size between 1, 0. Number of points for 0.05 = 20.
step = 0.05
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use Skranke
use_franke = True
#Max polynomial degree
maxDegree = 10
#Number of epochs
epochs = 200
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, step, maxDegree)
rho = 0.9
rho2 = 0.99
#hidden layer configuration gotten from optimize_nodes.py
adam = Adam(eta = 0.01 , rho = rho, rho2 = rho2)
adamom = AdagradMomentum(eta = 0.01, momentum = 0.5)
mom = Momentum(eta = 0.1, momentum= 0.5)
hidden_layer_sizes = (16, 16)
output_layer_size = 1
ffnn = FFNN((X.shape[1] , *hidden_layer_sizes, output_layer_size), hidden_func=sigmoid, seed=1984, output_func= lambda x: x)
batches = X.shape[0]

scores_adam = ffnn.cross_validation(X, z.reshape(-1, 1), folds, adam, batches, epochs, lam = 0.0001)
scores_adamom = ffnn.cross_validation(X, z.reshape(-1, 1), folds, adamom, batches, epochs, lam = 0.0001)


mlp_regressor = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='logistic',
    solver = 'adam',
    max_iter=epochs,
    alpha=0.0001,
    random_state=1984,
    batch_size = batches // 20,
)

mlp_regressor.fit(X_train, np.ravel(z_train))
mlp_scores = mlp_regressor.loss_curve_

plt.plot(scores_adam["val_errors"], label="Val set error FFNN Adam")
plt.plot(scores_adamom["val_errors"], label="Val set error FFNN Adagrad mom")
plt.plot(mlp_scores, label="scikit error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE as a function of Epochs for our FFNN and Scikit MLP regressor using Adam")
plt.show()

