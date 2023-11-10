from sklearn.datasets import load_breast_cancer
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from schedulers import *
from FFNN import *
from utils import *

seed = np.random.seed(4231)

cancer_dataset = load_breast_cancer()

X = cancer_dataset.data
z = cancer_dataset.target
z.reshape(z.shape[0], 1)


