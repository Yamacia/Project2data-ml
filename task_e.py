import autograd.numpy as np
from autograd import grad

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
#Number of datapoints to generate for
datapoints = 20
#Noise param for Franke function, use 0.0 for no noise
noise = 0.05
#If True use Franke, if False use cancer data
use_franke = False
#Max polynomial degree
maxDegree = 8
x, y, targets, inputs, X_train, X_test, z_train, z_test = generate_synth_dataset(use_franke, noise, 1 / datapoints, maxDegree)

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.zeros([inputs.shape[1]])
print(inputs)
print("Initial loss:", training_loss(weights))
for i in range(1):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))

design_matrix = np.ones([inputs.shape[1]]).reshape(-1,1)
# print(design_matrix)
print(inputs.shape)
print(np.concatenate((design_matrix, inputs), axis = 1))