# A sigmoid representation using ggplot for Python


from ggplot import *
import pandas as pd
import numpy as np

np.random.seed(314)

def sigmoid(x,y):
    """
    :param x:
    :param y:
    x.shape[-1] should be equal to y.shape[0].
    Otherwise, a the dot product can't be computed.
    :returns:  an array
    """
    try:
        return 1/(1 + np.exp(-np.dot(x,y)))
    except ValueError:
        return None


random_data = 2*np.random.random((100,2)) - 1

data = pd.DataFrame( data=random_data, columns=['x','y'])

print(data)
sigmoid_plot = ggplot(aes(x='y', y='y'), data=data) + geom_point()


print(sigmoid_plot)
