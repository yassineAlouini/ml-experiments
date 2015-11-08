# A sigmoid representation using ggplot for Python


from ggplot import *



def sigmoid(x,y):
    return 1/(1 + np.exp(-np.dot(x,y)))
