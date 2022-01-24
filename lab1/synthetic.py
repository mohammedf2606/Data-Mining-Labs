import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as r
from sklearn import datasets


x_rand, y_rand, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)
