
# coding: utf-8

# In[29]:

import pandas as pd
import matplotlib.pylab as plt
import io
import requests
import seaborn as sns
import pandas_profiling
import missingno as msno
get_ipython().magic('matplotlib inline')


# In[16]:

PROCESSED_DATA_URL = "https://raw.githubusercontent.com/BuzzFeedNews/H-2-certification-data/master/data/processed/H-2-certification-decisions.csv"


# In[19]:

s=requests.get(PROCESSED_DATA_URL).content
immigration_df=pd.read_csv(io.StringIO(s.decode('utf-8')))


# ## A general exploration of the immigration data

# In[21]:

immigration_df.head()


# In[31]:

immigration_report = pandas_profiling.ProfileReport(immigration_df)


# In[32]:

immigration_report.to_file('immigration_data_exploration_report.html')


# In[30]:

msno.matrix(immigration_df)


# In[ ]:



