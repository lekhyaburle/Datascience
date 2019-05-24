#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


ccpp = pd.read_excel('F:\\Analytics\\ML_Datasets\\CCPP\\ccpp.xlsx')


# In[3]:


ccpp.describe()


# In[4]:


ccpp.info()


# In[5]:


ccpp.head()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
ccpp.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()


# In[7]:


plt.scatter(ccpp['AT'],ccpp['PE'])


# In[8]:


plt.scatter(ccpp['V'],ccpp['PE'])


# In[9]:


plt.scatter(ccpp['AP'],ccpp['PE'])


# In[10]:


plt.scatter(ccpp['RH'],ccpp['PE'])


# In[11]:


corr_matrix = ccpp.corr()


# In[12]:


corr_matrix


# In[17]:


ccpp.plot(kind='box', subplots=True)
plt.show()


# In[16]:


plt.matshow(ccpp.corr())
plt.show()


# In[27]:


from pandas.plotting import scatter_matrix
scatter_matrix(ccpp)
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(ccpp[['AT','V','AP','RH']],ccpp['PE'], test_size=0.2, random_state=42)


# In[25]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[28]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()


# In[29]:


lin_reg.fit(X_train,y_train)


# In[30]:


lin_reg.predict(X_train)


# In[31]:


lin_reg.coef_


# In[33]:


lin_reg.intercept_


# In[39]:


lin_reg.score(X_train,y_train)


# In[41]:


plt.plot(X_train['AT'],lin_reg.predict(X_train),'r')
plt.scatter(X_train['AT'],y_train)


# In[42]:


from sklearn import metrics


# In[43]:


metrics.mean_squared_error(y_train,lin_reg.predict(X_train))


# In[44]:


metrics.r2_score(y_train,lin_reg.predict(X_train))


# In[45]:


metrics.mean_absolute_error(y_train,lin_reg.predict(X_train))


# In[46]:


plt.plot(X_test['AT'],lin_reg.predict(X_test),'r')
plt.scatter(X_test['AT'],y_test)


# In[47]:


metrics.mean_squared_error(y_test,lin_reg.predict(X_test))


# In[48]:


metrics.r2_score(y_test,lin_reg.predict(X_test))


# In[49]:


metrics.mean_absolute_error(y_test,lin_reg.predict(X_test))


# In[50]:


from sklearn.svm import SVR


# In[51]:


svr_reg = SVR()
svr_reg.fit(X_train, y_train)


# In[55]:


svr_reg.intercept_


# In[58]:


svr_reg.score(X_train,y_train)


# In[59]:


metrics.mean_squared_error(y_test,svr_reg.predict(X_test))


# In[60]:


metrics.r2_score(y_test,svr_reg.predict(X_test))


# In[61]:


metrics.mean_absolute_error(y_test,svr_reg.predict(X_test))


# In[62]:


plt.plot(X_test['AT'],svr_reg.predict(X_test),'r')
plt.scatter(X_test['AT'],y_test)


# In[63]:


svr_reg_c2 = SVR(C=2.0)
svr_reg_c2.fit(X_train, y_train)


# In[65]:


svr_reg_c2.score(X_train,y_train)


# In[66]:


metrics.mean_squared_error(y_test,svr_reg_c2.predict(X_test))


# In[67]:


metrics.r2_score(y_test,svr_reg_c2.predict(X_test))


# In[68]:


metrics.mean_absolute_error(y_test,svr_reg_c2.predict(X_test))


# In[70]:


plt.plot(X_test['AT'],svr_reg_c2.predict(X_test),'r')
plt.scatter(X_test['AT'],y_test)


# In[71]:


from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor()
sgd_reg.fit(X_train, y_train)


# In[72]:


sgd_reg.intercept_


# In[73]:


sgd_reg.coef_


# In[75]:


sgd_reg.score(X_train,y_train)


# In[76]:


metrics.mean_squared_error(y_test,sgd_reg.predict(X_test))


# In[77]:


plt.plot(X_test['AT'],sgd_reg.predict(X_test),'r')
plt.scatter(X_test['AT'],y_test)


# In[78]:





# In[ ]:




