
# coding: utf-8

# A notebook exploring basic **Theano** concepts and operations.
# This is mostly inspired from the following page: http://deeplearning.net/software/theano/tutorial/adding.html#adding-two-scalars

# In[2]:

import numpy as np
import theano.tensor as T
from theano import pp
from theano import function


# ## Create some variables and functions

# In[21]:

x = T.dscalar('x')
y = T.dscalar('y')
A = T.dmatrix('A')
B = T.dmatrix('B')
z = x + y
C = A * B
adding = function([x, y], z)
# Element wise multiplication of two matrices
multiply = function([A, B], C) 


# In[8]:

# Display x and y types
print(x.type, y.type)


# => `x` and `y` are **float** (64 to be more precise) **scalars** (tensors of dimension 0). 

# In[15]:

# Display A and B types
print(A.type, B.type)


# => A and B are **float** matrices (tensors of dimension 2)

# ## Basic computation using Theano tensors

# In[11]:

assert adding(2,3) == 5


# In[13]:

#Pretty print the variable z
pp(z)


# In[20]:

multiply(np.matrix([[1,1,1], [1,0,0]]), np.matrix([[1,1,1], [0,0,1]]))


# In[4]:

# Multiple operations can be performed at the same time
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
multiple_outputs = function([a, b], [diff, abs_diff, diff_squared])


# In[7]:

multiple_outputs(np.matrix([[1,1,1], [1,0,0]]), np.matrix([[1,1,1], [0,0,1]]))

