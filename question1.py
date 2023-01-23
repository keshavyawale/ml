#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


dfcar=pd.read_csv('data_1.csv')


# In[ ]:


#Data understanding and exploration


# In[4]:


#top 5 records 
dfcar.head()


# In[7]:


#shape of given data set is row=301 and column=9
shape=dfcar.shape
shape


# In[8]:


#understanding data type of columns and geting non null count 
dfcar.info()


# In[9]:


#understanding the central tendancy of given data set
dfcar.describe()


# In[40]:


sns.countplot(data=dfcar,x='Owner')


# In[12]:


#data set is not having null values
dfcar.isnull().sum()


# In[14]:


#year column having inverse relation with most of feature 
dfcar.corr()


# In[42]:


sns.countplot(data=dfcar,x='Car_Name')


# In[17]:


dfcar.nunique()


# In[44]:


sns.countplot(data=dfcar,x='Year')
plt.xticks(rotation=90)
plt.show()


# In[45]:


sns.countplot(data=dfcar,x='Transmission')


# In[23]:


#converting catagorical dato to numerical data (fule_type,seller_type,transmission)
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
j=dfcar
x=j.iloc[:,:-1]
x=x.apply(l.fit_transform)
x


# In[46]:


sns.heatmap(dfcar.corr(),annot=True)


# In[ ]:





# In[24]:


x.info()


# In[47]:


dfcar


# In[48]:


#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[49]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[50]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score
lg=LinearRegression()
model=lg.fit(x_train,y_train)


# In[51]:


y_pred=model.predict(x_test)


# In[52]:


model.score(x_test,y_test)


# In[53]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)


# In[54]:


y_pred=model.predict(x_test)


# In[55]:


y_pred


# In[56]:


model.score(x_test,y_test)


# In[ ]:




