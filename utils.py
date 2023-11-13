import numpy as np
import autograd.numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad
from sklearn.model_selection import train_test_split
from FFNN import *
from schedulers import *
from sklearn.datasets import load_breast_cancer


def FrankeFunction(x,y, noise = 0.0):
    if noise != 0.0:
        noise = noise * np.random.standard_normal(x.shape)
    

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise

def SkrankeFunction(x, y):
    return np.ravel(0 + 1*x + 2*y + 3*x**2 + 4*x*y + 5*y**2)

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def generate_dataset(use_franke, noise, step, maxDegree):
	x = np.arange(0, 1, step)
	y = np.arange(0, 1, step)
	x, y = np.meshgrid(x,y)
	if use_franke:
		z = FrankeFunction(x, y, noise)
		z = np.ravel(z)

		X =	create_X(x, y, maxDegree)

		X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

		z_train = z_train.reshape(z_train.shape[0], 1)
		z_test = z_test.reshape(z_test.shape[0], 1)
	else:
		#use the cancer data
		cancer = load_breast_cancer()
		z  = cancer.target
		X = cancer.data
		#Splitting data 4/5 train and 1/5 test, so more data to train than test
		X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2,random_state=0)
		
		z = z.reshape(-1,1)
		z_train = z_train.reshape(-1,1) 
		z_test = z_test.reshape(-1,1)

	return x, y, z, X, X_train, X_test, z_train, z_test
	
def optimize_n_hidden_layers(X, t, folds, scheduler, batches, epochs, lam, n_nodes, max_layers, hidden_func = sigmoid, output_func = lambda x :x, cost_func = CostOLS):
	scores_list = []
	attempted_layers = []
	for i in range(1, max_layers + 1):
		hidden_layer = (n_nodes,) * i
		ffnn = FFNN(dimensions=(X.shape[1], *hidden_layer, 1), hidden_func=hidden_func, seed=4231, output_func= output_func, cost_func= cost_func)
		scores = ffnn.cross_validation(X, t, folds, scheduler, batches, epochs, lam)
		scores_list.append(scores)
		attempted_layers.append(hidden_layer)
		print(f"\n Hidden layer: {hidden_layer}")
	return scores_list, attempted_layers

def optimize_n_nodes(X, t, folds, scheduler, batches, epochs, lam, hidden_layers, n_nodes, hidden_func = sigmoid, output_func = lambda x :x, cost_func = CostOLS):
	scores_list = []
	attempted_layers = []
	for nodes in (n_nodes):
		hidden_layer = (nodes,) * hidden_layers
		ffnn = FFNN(dimensions=(X.shape[1], *hidden_layer, 1), hidden_func=hidden_func, seed=4231, output_func= output_func, cost_func=cost_func)
		scores = ffnn.cross_validation(X, t, folds, scheduler, batches, epochs, lam)
		scores_list.append(scores)
		attempted_layers.append(hidden_layer)
		print(f"\n Hidden layer: {hidden_layer}")
	return scores_list, attempted_layers

def run_funcs(X, t, folds, batches, epochs, etas, lambdas, hidden_layers, hidden_funcs):
	scores_list = []
	i = 0
	for func in hidden_funcs:
		adam = Adam(etas[i], rho = 0.9, rho2 = 0.99)
		ffnn = FFNN(dimensions=(X.shape[1], *(hidden_layers[i]), 1), hidden_func=func, seed=4231, output_func= lambda x: x)
		scores = ffnn.cross_validation(X, t.reshape(-1, 1), folds, adam, batches, epochs, lambdas[i])
		scores_list.append(scores)
		i += 1
	return scores_list



