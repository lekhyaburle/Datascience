#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


pwd


# In[5]:


Notes_data = pd.read_csv('F:\\Analytics\\ML_Datasets\\bankNotes\\data_banknote_authentication.txt', header = None)


# In[17]:


Notes_data.head()


# In[6]:


Notes_data.describe()


# In[7]:


Notes_data.columns = ["Variance", "Skewness", "Kurtosis", "Entropy","Class"]


# In[8]:


Notes_data.describe()


# In[9]:


Notes_data.info()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
Notes_data.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


train_notes_set, test_notes_set = train_test_split(Notes_data, test_size=0.2, random_state=42)


# In[15]:


train_notes_set.shape


# In[16]:


test_notes_set.shape


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[19]:


plt.scatter(train_notes_set['Variance'],train_notes_set['Class'])


# In[20]:


plt.scatter(train_notes_set['Skewness'],train_notes_set['Class'])


# In[22]:


plt.scatter(train_notes_set['Kurtosis'],train_notes_set['Class'])


# In[23]:


plt.scatter(train_notes_set['Entropy'],train_notes_set['Class'])


# In[26]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[27]:


log_reg.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[29]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[30]:


svm_clf = Pipeline((
("scaler", StandardScaler()),
("linear_svc", LinearSVC(C=10, loss="hinge")),
))


# In[31]:


svm_clf.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[32]:


svm_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[33]:


from sklearn.metrics import confusion_matrix


# In[34]:


confusion_matrix(train_notes_set['Class'],svm_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[35]:


confusion_matrix(train_notes_set['Class'],log_reg.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[37]:


from sklearn.metrics import roc_curve


# In[38]:


fpr, tpr, thresholds = roc_curve(train_notes_set['Class'],svm_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr, tpr, thresholds


# In[39]:


fpr1, tpr1, thresholds1 = roc_curve(train_notes_set['Class'],log_reg.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr1, tpr1, thresholds1


# In[42]:


plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot(fpr1, tpr1, linewidth=2, label=None,color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[43]:


from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[106]:


tree_clf_woD = DecisionTreeClassifier()


# In[107]:


tree_clf_woD.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[108]:


tree_clf_woD.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[44]:


tree_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[109]:


from sklearn.tree import export_graphviz
export_graphviz(
                tree_clf_woD,
                out_file="bankNoteswoD.dot",
                feature_names=['Variance','Skewness','Kurtosis','Entropy'],
                class_names=['authentic','inauthentic'],
                rounded=True,
                filled=True
                )


# In[46]:


from sklearn.tree import export_graphviz
export_graphviz(
                tree_clf,
                out_file="bankNotesD2.dot",
                feature_names=['Variance','Skewness','Kurtosis','Entropy'],
                class_names=['authentic','inauthentic'],
                rounded=True,
                filled=True
                )


# In[47]:


confusion_matrix(train_notes_set['Class'],tree_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[48]:


fpr2, tpr2, thresholds2 = roc_curve(train_notes_set['Class'],tree_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr2, tpr2, thresholds2


# In[49]:


plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot(fpr1, tpr1, linewidth=2, label=None,color='red')
plt.plot(fpr2, tpr2, linewidth=2, label=None,color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[50]:


tree_clf3 = DecisionTreeClassifier(max_depth=3)
tree_clf3.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[51]:


export_graphviz(
                tree_clf3,
                out_file="bankNotesD3.dot",
                feature_names=['Variance','Skewness','Kurtosis','Entropy'],
                class_names=['authentic','inauthentic'],
                rounded=True,
                filled=True
                )


# In[52]:


confusion_matrix(train_notes_set['Class'],tree_clf3.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[53]:


fpr3, tpr3, thresholds3 = roc_curve(train_notes_set['Class'],tree_clf3.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr3, tpr3, thresholds3


# In[54]:


plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot(fpr1, tpr1, linewidth=2, label=None,color='red')
plt.plot(fpr2, tpr2, linewidth=2, label=None,color='green')
plt.plot(fpr3, tpr3, linewidth=2, label=None,color='yellow')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[55]:


log_reg.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[56]:


confusion_matrix(test_notes_set['Class'],log_reg.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[ ]:


random clssifier


# In[74]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])  


# In[75]:


clf.feature_importances_


# In[76]:


clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[60]:


clf.score(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']], test_notes_set['Class'])


# In[61]:


confusion_matrix(test_notes_set['Class'],clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[62]:


from sklearn.metrics import precision_score, recall_score


# In[63]:


precision_score(test_notes_set['Class'],clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[64]:


recall_score(test_notes_set['Class'],clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[65]:


fpr4, tpr4, thresholds4 = roc_curve(train_notes_set['Class'],clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr4, tpr4, thresholds4


# In[95]:


# plt.plot(fpr, tpr, linewidth=2, label=None)
# plt.plot(fpr1, tpr1, linewidth=2, label=None,color='red')
plt.plot(fpr2, tpr2, linewidth=2, label=None,color='green')
plt.plot(fpr3, tpr3, linewidth=2, label=None,color='yellow')
plt.plot(fpr4, tpr4, linewidth=2, label=None,color='blue')
plt.plot([0, 2], [0, 2], 'k--')
plt.axis([-0.5, 1, 0, 1.5])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[77]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
Grd_clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
Grd_clf.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])  


# In[78]:


Grd_clf.feature_importances_


# In[80]:


Grd_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[81]:


precision_score(test_notes_set['Class'],Grd_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[82]:


recall_score(test_notes_set['Class'],Grd_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[83]:


confusion_matrix(test_notes_set['Class'],Grd_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[84]:


fpr_grd, tpr_grd, thresholds_grd = roc_curve(train_notes_set['Class'],Grd_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr_grd, tpr_grd, thresholds_grd


# In[86]:


from sklearn import metrics


# In[87]:


metrics.auc(fpr4, tpr4)


# In[88]:


metrics.auc(fpr_grd, tpr_grd)


# In[89]:


metrics.auc(fpr2, tpr2)


# In[90]:


metrics.auc(fpr, tpr)


# In[93]:


recall_score(test_notes_set['Class'],Grd_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[96]:


from sklearn.naive_bayes import GaussianNB
Gaussian_clf = GaussianNB()
Gaussian_clf.fit(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']], train_notes_set['Class'])


# In[97]:


Gaussian_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[99]:


Gaussian_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']])


# In[101]:


confusion_matrix(test_notes_set['Class'],Gaussian_clf.predict(test_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))


# In[104]:


fpr_gaus, tpr_gaus, thresholds_gaus = roc_curve(train_notes_set['Class'],Gaussian_clf.predict(train_notes_set[['Variance','Skewness','Kurtosis','Entropy']]))
fpr_gaus, tpr_gaus, thresholds_gaus


# In[105]:


metrics.auc(fpr_gaus, tpr_gaus)


# In[ ]:




