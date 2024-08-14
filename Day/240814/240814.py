#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[6]:


x1 = np.random.choice([1,0],10,p=[0.2,0.8])
x2 = np.random.choice([1,0],10)


# In[9]:


x1 = np.where(x1==0,"cat","dog")


# In[10]:


x2 = np.where(x2==0,"cat","dog")


# $$ \sum_{k=1}^{n}p_{k}log(p_{k})$$

# In[15]:


entropy_x1 = -((9/10)*np.log2(9/10)+(1/10)*np.log2(1/10))


# In[21]:


entropy_x2 = -((6/10)*np.log2(6/10)+(4/10)*np.log2(4/10))


# In[22]:


## total entropy  - earn entropy = information gain
print(entropy_x1)
print(entropy_x2)


# In[23]:


import pandas as pd

# In[32]:


data.columns = ['outlook','temper','humidity','windy','play']


# In[34]:


data.to_csv('player.csv',index=None)


# In[36]:


player = pd.read_csv("player.csv")


# In[38]:


player.outlook.unique()


# In[58]:


def total_entropy(x):
    p = x.value_counts()/x.value_counts().sum()
    return -np.sum(p*np.log2(p))

total_entropy(player.play)-\
(total_entropy(player.loc[player.outlook == 'overcast','play']) +\
total_entropy(player.loc[player.outlook == 'rainy','play']) +\
total_entropy(player.loc[player.outlook == 'sunny','play']) )/3


# In[61]:


total_entropy(player.play)-\
(total_entropy(player.loc[player.temper == 'hot','play']) +\
total_entropy(player.loc[player.temper == 'cool','play']) +\
total_entropy(player.loc[player.temper == 'mild','play']) )/3


# In[63]:


total_entropy(player.play)-\
(total_entropy(player.loc[player.humidity == 'high','play']) +\
total_entropy(player.loc[player.humidity == 'normal','play']) )/2


# In[66]:


total_entropy(player.play)-\
(total_entropy(player.loc[player.windy == True,'play']) +\
total_entropy(player.loc[player.windy == False,'play']) )/2


# In[59]:


player.columns


# In[65]:


player.windy.dtype


# In[69]:


new_player = player.loc[player.outlook != 'overcast',:]


# In[70]:


new_player


# In[91]:


total_entropy(new_player.play)-\
(total_entropy(new_player.loc[new_player.outlook == 'rainy','play']) +\
total_entropy(new_player.loc[new_player.outlook == 'sunny','play']))/2


# In[92]:


total_entropy(new_player.play)-\
(total_entropy(new_player.loc[new_player.temper == 'cool','play']) +\
total_entropy(new_player.loc[new_player.temper == 'mild','play']) )/2


# In[93]:


total_entropy(new_player.play)-\
(total_entropy(new_player.loc[new_player.humidity == 'high','play']) +\
total_entropy(new_player.loc[new_player.humidity == 'normal','play']) )/2


# In[94]:


total_entropy(new_player.play)-\
(total_entropy(new_player.loc[new_player.windy == True,'play']) +\
total_entropy(new_player.loc[new_player.windy == False,'play']) )/2


# In[89]:


new_player = new_player.loc[new_player.temper != 'hot',:]


# In[96]:


new_player.loc[new_player.humidity == 'normal',"play"]


# In[98]:


new_player.loc[new_player.windy == False,"play"]


#%%
new_player

# ---------------------------------------------------------------------------------
#%%
from sklearn.naive_bayes import GaussianNB
# %%
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
# %%
def train_test(x):
    return x_tr, x_te, y_tr, y_te
# -------------------------------------------------------------
#%%
import pandas as pd
#%%
df = pd.read_csv("C:/Users/kimmi/OneDrive/바탕 화면/KEPCO/KEPCO_class content/T.Lee_SpringBoot,Python/Python/Study_ML_DL-1/Day/240814/players.csv")
# %%
features = df.isna()
#%%
df = df[features]
# %%
df = pd.read_csv("students.csv")
# %%
df = df.iloc[:,1:]
# %%
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
# %%
from sklearn.tree import DecisionTreeClassifier
# %%
tree_model = DecisionTreeClassifier()
tree_model.fit(x,y)
tree_model.score(x,y)
# %%
x_train, x_test, y_train, y_test = train_test(x,y)
# %%
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train,y_train)
tree_model.score(x_test,y_test)
# %%
from sklearn.svm import SVC
# %%
svm_model = SVC(C=1.0)
svm_model = 
# %%
svm_model.score()
# %%
svm_model.score()
# %%
from sklearn.linear_model import LogisticRegression
# %%
lm_model = Lo