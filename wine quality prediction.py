#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn


# In[2]:


df = pd.read_csv('C://Users//nisha//Downloads//wine//WineQualityPrediction-master//WineQualityPrediction-master//winequality-red.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


correlations = df.corr()['quality'].drop('quality')
print(correlations)


# In[6]:


sns.heatmap(df.corr())
plt.show()


# In[7]:


def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations


# In[8]:


features = get_features(0.05)
print(features)
x = df[features]
y = df['quality']


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)


# In[10]:


# x_train.shape
# x_test.shape
# y_train.shape
y_test.shape


# In[11]:


# fitting linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[12]:


regressor.coef_


# In[13]:


train_pred = regressor.predict(x_train)
train_pred


# In[14]:


test_pred = regressor.predict(x_test)
test_pred


# In[18]:


train_rmse = sklearn.metrics.mean_squared_error(train_pred, y_train)
train_rmse


# In[20]:


test_rmse =sklearn.metrics.mean_squared_error(test_pred, y_test) ** 0.5
test_rmse


# In[21]:


predicted_data = np.round_(test_pred)
predicted_data


# In[22]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# In[23]:


coeffecients = pd.DataFrame(regressor.coef_,features)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




