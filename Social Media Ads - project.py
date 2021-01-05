#!/usr/bin/env python
# coding: utf-8

# # Day10 - Python

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


os.chdir("D:\\Imarticus\\Python Cls\\python prjts\\data\\Classification problem")


# In[4]:


ad_data= pd.read_csv("Social_Network_Ads.csv")


# In[5]:


ad_data.head()


# In[6]:


x= ad_data.iloc[:,2:4].values
y= ad_data.iloc[:,4].values


# In[7]:


x


# In[8]:


y


# In[9]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x= sc_x.fit_transform(x)


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state=40)


# In[39]:


#import package for KNN
from sklearn.neighbors import KNeighborsClassifier


# In[40]:


# instantiate knn classifier
model = KNeighborsClassifier()


# In[41]:


model.fit(x_train,y_train)


# In[42]:


y_pred_kn= model.predict(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_kn)


# In[44]:


cm


# In[45]:


(53+21)/(53+3+3+21)


# # Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[47]:


y_pred_log=logmodel.predict(x_test)


# In[48]:


cm1 = confusion_matrix(y_test,y_pred_log)


# In[49]:


cm1


# In[50]:


(54+13)/(54+2+11+13)


# # Naive Bayes Theorm

# In[51]:


#IMPORTING NAIVEBAYES THEORM
from sklearn.naive_bayes import GaussianNB

# Creat a Gaussian Classifier
nb_mdl = GaussianNB()

#train the model using the training set
nb_mdl.fit(x_train,y_train)


# In[52]:


#predict the response for test dataset
y_pred_nb = nb_mdl.predict(x_test)


# In[53]:


cm2 = confusion_matrix(y_test,y_pred_nb)


# In[54]:


cm2


# In[55]:


(54+19)/(54+2+5+19)


# In[29]:


#XG Boost Method


# In[56]:


from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(x_train,y_train)


# In[57]:


y_pred_xb = classifier_xgb.predict(x_test)


# In[58]:


cm3= confusion_matrix(y_test,y_pred_xb)
cm3


# In[59]:


(53+20)/(53+3+4+20)


# # K - Fold method for XGBoost

# In[64]:


from sklearn.model_selection import cross_val_score
accuracy_xgb= cross_val_score(estimator= classifier_xgb, X=x_train,y=y_train, cv=10)
accuracy_xgb


# In[65]:


accuracy_xgb.mean()


# # K - Fold method for NaiveBayes Theorm

# In[66]:


from sklearn.model_selection import cross_val_score
accuracy_nb= cross_val_score(estimator=nb_mdl , X=x_train,y=y_train, cv=10)
accuracy_nb


# In[68]:


accuracy_nb[3]


# # K - Fold method for Logistic Regression

# In[69]:


from sklearn.model_selection import cross_val_score
accuracy_lg= cross_val_score(estimator=logmodel , X=x_train,y=y_train, cv=10)
accuracy_lg


# # K - Fold method for KNN method

# In[72]:


from sklearn.model_selection import cross_val_score
accuracy_knn = cross_val_score(estimator=model , X=x_train,y=y_train, cv=10)
accuracy_knn


# In[73]:


accuracy_knn.mean()


# In[ ]:




