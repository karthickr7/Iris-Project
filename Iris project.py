#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[24]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


# # Load Dataset

# In[8]:


iris=datasets.load_iris()


# In[9]:


iris


# # Features

# In[10]:


print(iris.feature_names)


# In[11]:


print(iris.target_names)


# In[12]:


iris.data


# In[13]:


iris.target


# # Assigning Input and Output 

# In[72]:


X=iris.data


# In[73]:


Y=iris.target


# In[74]:


X.shape


# In[75]:


Y.shape


# In[76]:


ml=RandomForestClassifier()


# In[77]:


ml.fit(X, Y)


# # Feature Importance

# In[23]:


print(ml.feature_importances_)


# In[25]:


plt.barh(iris.feature_names, ml.feature_importances_)


# # Prediction

# In[26]:


X[0]


# In[28]:


print(ml.predict([[5.1, 3.5, 1.4, 0.2]]))


# In[32]:


print(ml.predict(X[[0]]))


# In[33]:


print(ml.predict_proba(X[[0]]))


# # Data Split (75/25)

# In[62]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


# In[63]:


X_train.shape, Y_train.shape


# In[64]:


X_test.shape, Y_test.shape


# # Model Rebuilding

# In[65]:


#
ml.fit(X_train, Y_train)


# In[66]:


# Predictions on a sample from the dataset
print(ml.predict([[5.1, 3.5, 1.4, 0.2]]))


# In[67]:


# Probability prediction
print(ml.predict_proba([[5.1, 3.5, 1.4, 0.2]]))


# In[68]:


# Predictions on Test dataset
print(ml.predict(X_test))


# In[69]:


print(Y_test)


# # Model performance

# In[71]:


#'Score' measures how many models a Random Forest model has got it right
print(ml.score(X_test, Y_test))


# In[ ]:





# In[ ]:




