# A sigmoid representation using ggplot for Python


import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from ggplot import *

np.random.seed(314)


def sigmoid(x, y):
    """
    :param x:
    :param y:
    x.shape[-1] should be equal to y.shape[0].
    Otherwise, a the dot product can't be computed.
    :returns:  an array
    """
    try:
        return 1 / (1 + np.exp(-np.dot(x, y)))
    except ValueError:
        return None


def main():

    # Generate random data and store it in a pandas data frame
    random_data = 2 * np.random.random((100, 2)) - 1
    data = pd.DataFrame(data=random_data, columns=['x', 'y'])

    # Add the sigmoid function output applied to the x and y columns
    data['z'] = sigmoid(data['x'], data['y'])
    print(data.head())

if __name__ == '__main__':
    main()
