#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("train.csv")
train.head()


# In[2]:


train.count()


# In[3]:


sns.countplot(x='Survived', hue='Pclass', data=train)


# In[4]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train["Sex"]=encoder.fit_transform(train[["Sex"]])
train


# In[6]:


train.drop("Cabin",inplace=True,axis=1)


# In[7]:


train.dropna(inplace=True)


# In[8]:


train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)


# In[9]:


X = train.drop("Survived",axis=1)
y = train["Survived"]


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[11]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[12]:


predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

