
# coding: utf-8

# 
# In this notebook, I try to implement pandas plotting functions using basic matplotlib functions.
# * Author: Yassine Alouini
# * Date: 7-5-2016
# * License: MIT

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:

# The recent grads dataset is assumed to be at the same level 
# as the notebook
recent_grads_df = pd.read_csv("recent-grads.csv")


# In[8]:

# First line of the recent grads data
recent_grads_df.head(1)


# In[9]:

# Last line of the recent grads data
recent_grads_df.tail(1)


# In[11]:

# Some general descriptive statistics
recent_grads_df.describe()


# In[14]:

# Looking for missing values and counting the rows
cleaned_recent_grads_df = recent_grads_df.dropna()
removed_rows_count = len(recent_grads_df) - len(cleaned_recent_grads_df)
print("{line} line(s) removed".format(line=removed_rows_count))


# In[80]:

# A scatter matrix using pandas plotting tools
from pandas.tools.plotting import scatter_matrix
scatter_matrix(cleaned_recent_grads_df[
        ["ShareWomen","Unemployment_rate"]])


# In[53]:

# The same plot as above, this time using matplotlib 
# low-level functions.
fig, axes = plt.subplots(2,2, figsize=(10,10))
axes = axes.ravel()
ax1, ax2, ax3, ax4 = axes


cleaned_recent_grads_df["ShareWomen"].hist(ax=ax1)
cleaned_recent_grads_df.plot(x='Unemployment_rate',
                                     y='ShareWomen',
                                     ax=ax2, kind='scatter')
cleaned_recent_grads_df.plot(x='ShareWomen', 
                                     y='Unemployment_rate', 
                                     ax=ax3, kind='scatter')
cleaned_recent_grads_df["Unemployment_rate"].hist(ax=ax4)

# Remove some x and y axes
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)

# Set axes labels

ax1.set_ylabel("ShareWomen")
ax3.set_xlabel("ShareWomen")
ax3.set_ylabel("Unemployment_rate")
ax4.set_xlabel("Unemployment_rate")

# Set axes tick labels

ax1.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
ax3.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8], rotation=90)
ax3.set_yticklabels([0.00, 0.05, 0.10, 0.15])
ax4.set_xticklabels([0.00, 0.05, 0.10, 0.15, 0.20], rotation=90)

# Adjust x and y axes value limits

ax1.set_ylim(0,30)
ax2.set_xlim(0.0, 0.20)
ax3.set_xlim(0.0, 1.0)
ax3.set_ylim(0.0, 0.20)
ax4.set_xlim(0.0, 0.20)

# Adjust figure spacing

fig.subplots_adjust(wspace=0, hspace=0)


# In[54]:

# Create a ShareMen column containing the proportion of men
recent_grads_df["ShareMen"] = recent_grads_df["Men"] / recent_grads_df["Total"]


# In[70]:

# Select only the "Arts" majors
arts_grads_df = recent_grads_df[recent_grads_df.Major_category == "Arts"]


# In[78]:

# A stacked bar plot using pandas
arts_grads_df.set_index("Major")[["ShareMen", "ShareWomen"]].plot(figsize=(8,8), 
                                                                  kind="bar")


# In[76]:

# The same plot as above, this time using matplotlib 
# low-level functions.
import numpy as np
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
count_majors = len(arts_grads_df["Major"].unique())
locs = np.arange(count_majors)
bar_1 = ax.bar(left=locs, 
               height=arts_grads_df["ShareMen"].tolist(), 
               width=0.35)
ax.set_xticklabels(arts_grads_df["Major"].tolist(), rotation=90)
offset_locs = locs + 0.35
bar_2 = ax.bar(left=offset_locs, 
               height=arts_grads_df["ShareWomen"].tolist(), 
               width=0.35, color="green")
ax.set_xticks(offset_locs)
ax.legend((bar_1, bar_2), ("ShareMen", "ShareWomen"), loc="upper left")
plt.grid()

