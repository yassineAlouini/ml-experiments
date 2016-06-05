import numpy as np
from matplotlib import pylab as plt

np.random.seed(31415926)  # Set a seed for reproducibility

N_STEPS = 1000
random_data = np.random.randint(0, 2, N_STEPS)
# A symmetric random walk
random_walk = np.where(random_data > 0, 1, -1).cumsum()


N_WALKS = 1000
random_data_matrix = np.random.randint(0, 2, size=(N_WALKS, N_STEPS))

# Multiple symmetric random walks
multiple_random_walks = np.where(random_data_matrix > 0, 1, -1).cumsum(axis=1)
