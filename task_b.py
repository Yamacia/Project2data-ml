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
#Beta values 
betas = 10
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use Skranke
use_franke = True
#Max polynomial degree
maxDegree = 10
#Number of epochs
epochs = 500
folds = 5
#Generates either Skranke or Franke dataset
x, y, z, X, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, step, maxDegree)
eta = 0.001
lam = 0.01
rho = 0.9
rho2 = 0.99

scheduler = Adam(eta, rho, rho2)

hidden_layer_sizes = (60, 60, 60)
output_layer_size = 1
ffnn = FFNN((X.shape[1] , *hidden_layer_sizes, output_layer_size), hidden_func=sigmoid, seed=1984)


scores = ffnn.cross_validation(X, z.reshape(-1, 1), folds, scheduler, 1, epochs, lam)

mlp_regressor = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='logistic',
    solver = 'adam',
    max_iter=epochs,
    alpha=lam,
    random_state=1984,
    batch_size= 1,
)

mlp_regressor.fit(X_train, np.ravel(z_train))
mlp_scores = mlp_regressor.loss_curve_

plt.plot(scores["train_errors"], label="Train set error FFNN")
plt.plot(scores["val_errors"], label="Val set error FFNN")
plt.plot(mlp_scores, label="scikit error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE as a function of Epochs for our FFNN and Scikit MLP regressor")
plt.show()