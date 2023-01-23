#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 2:
# 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_excel('data_final.xlsx')


# In[3]:


dataset


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset.shape


# In[7]:


dataset.nunique()


# In[8]:


x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
x=x.reshape(-1,1)


# In[9]:


x


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[12]:


from sklearn.preprocessing import PolynomialFeatures
p=PolynomialFeatures(degree=4)
x_poly=p.fit_transform(x)


# In[13]:


x_poly


# In[14]:


from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(x,y)


# In[15]:


from sklearn.linear_model import LinearRegression
lr2=LinearRegression()
lr2.fit(x_poly,y)


# In[16]:


plt.scatter(x,y,color='red')
plt.plot(x,lr1.predict(x),color='blue')


# In[17]:


plt.scatter(x,y,color='red')
plt.plot(x,lr2.predict(x_poly),color='blue')


# In[ ]:




