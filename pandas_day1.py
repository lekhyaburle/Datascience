#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


path = "F:/Analytics/Python/Pandas/Class/data/"


# In[3]:


import pandas as pd


# In[4]:


credit_df=pd.read_csv(path+"Credit.csv")


# In[5]:


type(credit_df)


# In[6]:


# dir(credit_df)


# In[6]:


credit_df.head()


# In[7]:


credit_df.tail()


# In[10]:


credit_df.dtypes


# In[11]:


credit_df.columns


# In[12]:


credit_df.index


# In[20]:


credit_df.describe()


# In[14]:


len(credit_df)


# In[15]:


credit_df.size


# In[16]:


credit_df.shape


# In[18]:


credit_df.info()


# In[19]:


credit_df.memory_usage()


# In[9]:


age_series=credit_df["Age"]
age_series


# In[11]:


income_series=credit_df["Income"]
marital_series=credit_df["Married"]


# In[12]:


marital_series.unique()


# In[25]:


income_series


# In[26]:


ed_series=credit_df["Education"]


# In[27]:


ed_series


# In[28]:


ed_series.unique()


# In[31]:


sorted(list(ed_series.unique()))


# In[33]:


credit_df.columns


# In[35]:


df=credit_df.drop(columns=['Unnamed: 0','Education'])


# In[36]:


df.columns


# In[66]:


credit_df.set_index('Unnamed: 0',inplace=True)


# In[59]:


# credit_df.rename(columns={"Married" : "Marital Status"},inplace=True)


# In[67]:


credit_df.rename_axis("",inplace=True)


# In[57]:


df.head()


# In[68]:


credit_df.head()


# In[69]:


credit_df.loc[2:6,["Age","Married","Income"]]


# In[64]:


df.iloc[1:5,[0,4,7]]


# In[84]:


df[((df["Age"] <= 30) & (df["Married"]=="Yes"))]


# In[85]:


df_30 = df[(df["Age"] <= 30)]


# In[86]:


df_30["Age"].unique()


# In[89]:


df_30.head()


# In[116]:


df_groupby = df_30.groupby(["Age","Gender","Married"])["Income"].min()


# In[ ]:


df_groupby = df_30.groupby(["Age","Gender","Married"])["Income"].agg


# In[117]:


type(df_groupby)


# In[113]:


df_groupby.reset_index()


# In[ ]:




