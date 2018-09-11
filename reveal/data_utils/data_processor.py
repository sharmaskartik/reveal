import numpy as np

def normalize(x):
    return (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
