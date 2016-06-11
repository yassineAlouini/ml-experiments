
# coding: utf-8

# This notebook is the implementation of the [Welch Labs](http://www.welchlabs.com/) excellent video tutorials series (links inside the notebook below)

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from IPython.display import YouTubeVideo
get_ipython().magic('matplotlib inline')


# # Tutorial videos

# ## Part I: NN introduction
# 

# In[37]:

YouTubeVideo("5MXp9UUkSmc", width=600, height=400)


# ## Part II: NN definition

# In[33]:

YouTubeVideo("JLTXatV5dc0", width=600, height=400)


# ## Part III: Cost function and the curse of dimensionality

# In[32]:

YouTubeVideo("2igVhuRFOZM", width=600, height=400)


# ## Part IV: Gradient descent and the chain rule

# In[36]:

YouTubeVideo("O-Lc26lZpKU", width=600, height=400)


# # A representation of the sigmoid function and its derivative 

# ## Sigmoid 

# In[2]:

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# ## Derivative sigmoid 

# In[3]:

# The derivative of the sigmoid function
def sigmoid_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


# ## Plots

# In[4]:

x = np.arange(-10, 10, 0.01)
y = sigmoid(x)
y_prime = sigmoid_prime(x)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(x, y)
ax.plot(x, y_prime)
ax.set_title('A sigmoid function and its derivative')
ax.legend(['sigmoid', 'derivative sigmoid'], loc='best')
sns.despine()


# # Neural network class

# In[7]:

class NeuralNetwork(object):
    def __init__(self):
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3
        self.W_1 = np.random.rand(self.input_layer_size, self.hidden_layer_size)
        self.W_2 = np.random.rand(self.hidden_layer_size, self.output_layer_size)
    def forward(self, X):
        self.z_2 = np.dot(X, self.W_1)
        self.a_2 = sigmoid(self.z_2)
        self.z_3 = np.dot(self.a_2, self.W_2)
        y_hat = sigmoid(self.z_3)
        return y_hat
    def cost_function(self, X, y):
        return 0.5 * ((self.forward(X) - y) ** 2).mean()
    def cost_function_prime(self, X, y):
        self.y_hat = self.forward(X)
        error = (y- self.yhat)
        delta_3 = np.multiply(-error, sigmoid_prime(self.z_3))
        dJ_dW_2 = np.dot(self.a_2.T, delta_3)
        delta_2 = np.dot(delta_3, self.W_2.T) * sigmoid_prime(self.z_2)
        dJ_dW_1 = np.dot(X.T, delta_2)
        return dJ_dW_1, dJ_dW_2

