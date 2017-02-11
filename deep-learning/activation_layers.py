
# coding: utf-8

# A notebook showing how some very common activation functions (used for deep-learning for example) look like. Enjoy!

# In[20]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


# In[16]:

sns.set(font_scale=1.5)


# # Define the activiation metrics

# In[39]:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[8]:

def tanh(x):
    return np.tanh(x)


# In[13]:

def relu(x):
    return np.maximum(x, 0) # element-wise maximum


# # Plot them

# In[52]:

class ActivationPlots(object):
    def __init__(self, metrics):
        self.x = np.arange(-10, 10, 0.1)
        self.metrics = metrics
        self.n_plots = len(self.metrics)
    def build(self, axes):
        for ax, metric in zip(axes, self.metrics):
            y = metric(self.x)
            ax.plot(self.x, y)
            ax.set_title(str(metric.__name__))
        return axes
    def plot(self):
        n_rows = self.n_plots % 2
        n_cols = int(self.n_plots / n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4))
        self.build(axes)


# In[53]:

ActivationPlots([sigmoid, tanh, relu]).plot()

