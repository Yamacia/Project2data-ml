import numpy as np
import autograd.numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score
from autograd import grad
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y, noise = 0.0):
    if noise != 0.0:
        noise = np.random.normal(0, noise, x.shape)
    

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

def generate_synth_dataset(use_franke, noise, step, maxDegree):
	x = np.arange(0, 1, step)
	y = np.arange(0, 1, step)
	if use_franke:
		z = FrankeFunction(x, y, noise)
	else:
		z = SkrankeFunction(x, y)

	z = np.ravel(z)

	X =	create_X(x, y, maxDegree)

	X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

	z_train = z_train.reshape(z_train.shape[0], 1)
	z_test = z_test.reshape(z_test.shape[0], 1)

	return x, y, z, X, X_train, X_test, z_train, z_test
	
def min_max_scaler(data):
    """
    Perform Min-Max scaling on a given dataset.

    Min-Max scaling rescales the values in the input array 'X' to a given range
    .
    Parameters:
    x (np.array): The input dataset to be scaled.

    Returns:
    scaled(np.array): The scaled dataset with values in the range [0, 1].
    """
    min_i = min(data)
    max_i = max(data)
    
    scaled = (data-min_i)/(max_i-min_i)
    return scaled