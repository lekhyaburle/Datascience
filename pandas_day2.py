#!/usr/bin/env python
# coding: utf-8

# In[1]:


array=[['one','two','three','four','five'],['bar','foo','baz','quix',
                                           'tax']]


# In[2]:


l1=['one','two','three','four','five','one','two','three','five']


# In[3]:


l2=['bar','foo','baz','quix','tax','tax','bar','foo','fox']


# In[4]:


tuples=tuple(zip(l1,l2))


# In[5]:


print (tuples)


# In[6]:


import pandas as pd


# In[7]:


import numpy as np
np.random.seed(1234)


# In[8]:


np.random.randn(8)


# In[9]:


np.random.rand(8)


# In[10]:


np.random.random(3)


# In[11]:


np.random.randn(4,4)


# In[12]:


np.random.rand(4,4)


# In[13]:


np.random.random(16).reshape(4,4)


# In[14]:


df=pd.DataFrame(np.random.randn(5,3),columns=['one','two','three'],index=['a','b','d','f','h'])
df


# In[15]:


df_copied=df.reindex(['a','b','c','d','e','f','g','h'])
df_copied


# In[16]:


df_copied['Four']='Fox'


# In[17]:


df_copied


# In[18]:


df_copied['Five']= np.nan


# In[19]:


df_copied


# In[20]:


type(df_copied.loc["c","one"])


# In[21]:


pd.isnull(df_copied)


# In[22]:


df_nan=df_copied[(df_copied["Five"]!=np.nan)]
df_nan


# In[23]:


df_copied["Five"]!=np.nan


# In[24]:


df_copied[df_copied["one"].isnull()].any(axis=1)


# In[25]:


dir(df_copied["one"])


# In[26]:


df_copied


# In[27]:


df_copied.dropna(axis=1)
# df_copied


# In[28]:


df_copied.drop('Five',axis=1,inplace=True)


# In[29]:


df_copied.dropna()


# In[30]:


df_copied=df_copied.reindex(['a','b','c','c1','d','e','e1','e2','f','g','g1','g2','g3','h'])


# In[31]:


df_copied


# In[32]:


df_copied['Four']='Bat'


# In[33]:


df_copied.fillna(method='pad')


# In[34]:


df_copied.fillna(method='bfill')


# In[35]:


df_copied.fillna(method='pad',limit=3)


# In[36]:


df_copied.fillna(method='bfill',limit=2)


# In[37]:


df_copied.describe()


# In[38]:


df_copied.fillna(df_copied.mean())


# In[188]:


get_ipython().run_line_magic('pinfo', 'df_copied.fillna')


# In[39]:


x=np.arange(10)
x


# In[40]:


a=np.linspace(0,20,25)
a


# In[41]:


a**2


# In[42]:


from math import *


# In[43]:


sin(a[5])


# In[44]:


df_copied["Sine"] = df_copied['one'].apply(lambda x:sin(x))


# In[45]:


df_copied


# In[46]:


df_interpolate = df_copied.interpolate()


# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


plt.figure()


# In[49]:


plt.plot(df_interpolate.index,df_interpolate['two'])


# In[ ]:





# In[50]:


df_copied


# In[53]:


df_copied['one'].interpolate(method='piecewise_polynomial')


# In[52]:


df_copied["one"] = pd.to_numeric(df_copied["one"], errors='coerce')


# In[240]:


get_ipython().run_line_magic('pinfo', 'df_copied.interpolate')


# In[ ]:




