#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# In[3]:


X, y = load_wine().data, load_wine().target
X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[4]:


tree_model = DecisionTreeClassifier()
tree_model.fit(X_train,y_train)


# In[5]:


plot_tree(tree_model)


# In[10]:


## hyper-parameter
hyper_parameter = {"max_depth":3,"criterion":'entropy'}
tree_model = DecisionTreeClassifier(**hyper_parameter)


# In[11]:


tree_model.fit(X_train,y_train)


# In[12]:


tree_model.score(X_test,y_test)


# In[20]:


from sklearn.pipeline import Pipeline


# In[25]:


from sklearn.decomposition import PCA


# In[29]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

model_pipe = Pipeline([('pca', PCA(2)),
                        ('scaler', StandardScaler()), 
                       ('svc', SVC())])


# In[34]:


sscaler = StandardScaler()
sscaler.fit(X_train)
X_train_std = sscaler.transform(X_train)

pca = PCA(2)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)

svc_model = SVC(probability=True)
svc_model.fit(X_train_pca,y_train)


# In[37]:


X_test_std = sscaler.transform(X_test)
X_test_pca = pca.transform(X_test_std)
svc_model.predict(X_test_pca)


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
svc_model = Pipeline([
    ('sclaing',StandardScaler()),
    ('pca',PCA(3)),
    ('neighbor_estimator',KNeighborsClassifier())
])


# In[40]:


X_train.shape


# In[41]:


svc_model.fit(X_train,y_train)


# In[42]:


svc_model.score(X_test,y_test)


# In[43]:


svc_model = SVC()
svc_model.get_params()


# In[71]:


## kernel 'rbf', 'sigmoid', 'linear'
## C : 0.1 ~ 2 [20]
kernel = ['rbf','sigmoid','linear']
C = np.linspace(0.1,2,20)
scores = []
params = []
for c in C:
    for k in kernel:
        svc_model = SVC(kernel=k,C=c)
        svc_model.fit(X_train,y_train)
        score = svc_model.score(X_test,y_test)
        scores.append(score)
        params.append({'kernel':k,'C':c,})
best_param = params[np.argmax(np.array(scores))]
best_score = np.max(np.array(scores))


# In[74]:


svc_model = SVC(**best_param)


# In[75]:


svc_model.fit(X_train,y_train)
svc_model.score(X_test,y_test)


# In[70]:


scores[np.argmax(np.array(best_score))]


# In[78]:


from sklearn.model_selection import GridSearchCV


# In[82]:


param_grid = {
    'C':np.linspace(0.1,2,20),
    'kernel':['rbf','sigmoid','linear'] }
grid_search_model = GridSearchCV(SVC(),param_grid=param_grid,cv=5)


# In[84]:


grid_search_model.fit(X_train,y_train)


# In[86]:


grid_search_model.best_score_


# In[87]:


grid_search_model.best_params_


# In[91]:


grid_search_model.best_estimator_.score(X_test,y_test)


# In[ ]:





# In[92]:


dir(grid_search_model)


# In[281]:


class GridSearchCrossVal:
    def __init__(self,estimator,param_grid,cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_estimator_ = None
        self.best_params_ =  {}
        self.best_scores_ = 0
    def fit(self,x,y):
        param_list = list(product(*self.param_grid.values()))
        self.scores = []
        self.params = []
        for val in param_list:
            param = dict(zip(self.param_grid.keys(),val))
            self.params.append(param)
            self.estimator.set_params(**param)
            score = 0
            for i in range(self.cv):
                X_train, X_test, y_train, y_test = train_test_split(x,y)
                self.estimator.fit(X_train,y_train)
                score += self.estimator.score(X_test,y_test)
            score = score/self.cv
            self.scores.append(score)
        self.best_score_ = np.max(self.scores)
        self.best_params_ = self.params[np.argmax(np.array(self.scores))]
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
        self.best_estimator_.fit(x,y)
        return self.best_score_
    
    def predict(self,x):
        return self.best_estimator_.predict(x)
    
    def score(self,x,y):
        y_hat = self.predict(x)
        return np.sum(y == y_hat)/y.size
    


# In[282]:


param_grid = {
    'C':np.linspace(0.1,2,20),
    'kernel':['rbf','linear','sigmoid'],
    'probability':[True,False]
}
gsc = GridSearchCrossVal(SVC(),param_grid,7)


# In[283]:


gsc.fit(X_train,y_train)


# In[284]:


gsc.score(X_test,y_test)


# In[271]:


gsc.predict(X_test)


# In[261]:


gsc.best_estimator_.score(X_test,y_test)


# In[133]:


params = []
for key in param_grid:
    for val in param_grid[key]:
        params.append({key:val})


# In[159]:


get_ipython().run_line_magic('pinfo2', 'zip_longest')


# In[174]:


from itertools import combinations, product
a = ['a','b']
b = [0.3,0.2]
list(combinations([0.2,'linear','rbf'],2))


# In[184]:


len(list(product(list(param_grid.values())[0],list(param_grid.values())[1])))


# In[178]:


list(product(param_grid.values,['rbf','sigmoid']))


# In[171]:


from itertools import zip_longest,combinations
(list(zip_longest(*param_grid.values(),fillvalue=['rbf','linear','sigmoid'])))


# In[150]:


import itertools

list(itertools.combinations(params,2))
params


# In[125]:


p


# In[285]:


from sklearn.datasets import load_iris


# In[286]:


X = load_iris().data
y = load_iris().target


# In[291]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[296]:


model1 = KNeighborsClassifier()
model2 = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.6)


# In[297]:


model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
print(model1.score(X_test,y_test))
print(model2.score(X_test,y_test))


# In[299]:





# In[301]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[304]:


## 군집화 비지도학습 , train, test 를 분리할 필요가 없음
km = KMeans(n_clusters=3)
X_std = StandardScaler().fit_transform(X)
km.fit(X_std)


# In[307]:


km.labels_


# In[310]:


km.cluster_centers_


# In[314]:


km1 = KMeans(3)
km1.fit(X)


# In[315]:


km1.cluster_centers_


# In[341]:


km1.predict(X[0].reshape(1,-1))


# In[342]:


np.argmin(np.sum(np.square(km1.cluster_centers_ - X[0].reshape(1,-1)),1))


# In[333]:





# In[343]:


df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")


# In[346]:


df1 = df[['Sex','SibSp','Pclass']]


# In[347]:


from sklearn.preprocessing import LabelEncoder


# In[348]:


new_gender = LabelEncoder().fit_transform(df1.Sex)


# In[350]:


df1.drop(columns=['Sex'],inplace=True)


# In[351]:


df1['new_gender'] = new_gender


# In[353]:


km = KMeans(3)


# In[356]:


km.fit(df1)


# In[358]:


df['cluster1'] = km.labels_


# In[366]:


df.groupby(['cluster1']).Age.mean()


# In[371]:


df.loc[(df.Age.isna()) & (df.cluster1 == 2) ,'Age'] = df.groupby(['cluster1']).Age.mean()[2]


# In[372]:


df.info()


# In[523]:


class Kcluster:
    def __init__(self,n_cluseters=3):
        self.n_clusters = n_clusters
    def fit(self,x):
        x = StandardScaler().fit_transform(x)
        self.cluster_centers_ = np.random.randn(n_clusters,x.shape[1])
        for _ in range(x.shape[0]):
            for i in range(n_clusters):
                pos = np.argmin(np.sum(np.square(x - self.cluster_centers_[i]),1))
                self.cluster_centers_[i] = (x[pos] + self.cluster_centers_[i])/2
        self.labels_ = self.predict(x)
    def predict(self,x):
        x = StandardScaler().fit_transform(x)
        f = lambda x : np.argmin(np.sum(np.square(x-km.cluster_centers_),1))
        return np.apply_along_axis(f,1,x) 


# In[537]:


km = Kcluster(3)


# In[538]:


X = load_iris().data
km.fit(X)


# In[539]:


y = load_iris().target
y


# In[540]:


km.labels_


# In[541]:


kk = KMeans(3)
kk.fit(X)
km.cluster_centers_ = kk.cluster_centers_


# In[543]:


from sklearn import cluster


# In[545]:


from sklearn.cluster import AgglomerativeClustering


# In[555]:


hcluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None)


# In[557]:


hcluster.fit(X)


# In[558]:


from scipy.cluster.hierarchy import dendrogram


# In[561]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[562]:


plot_dendrogram(hcluster)


# In[563]:


from sklearn import datasets


# In[565]:


dir(datasets)


# In[580]:


from sklearn.datasets import make_circles, make_blobs, make_moons


# In[569]:


get_ipython().run_line_magic('pinfo2', 'make_circles')


# In[591]:


X, y = make_circles(100,factor=0.3, noise=0.05)


# In[592]:


import matplotlib.pyplot as plt


# In[594]:


plt.scatter(X[:,0],X[:,1],c=y)


# In[595]:


km = KMeans(2)
km.fit(X)


# In[597]:


plt.scatter(X[:,0],X[:,1],c=y)


# In[598]:


from sklearn.cluster import DBSCAN


# In[608]:


model = DBSCAN(eps=0.5)
model.fit(X)


# In[610]:


for eps in np.linspace(0.01,1.0,10):
    model = DBSCAN(eps=eps)
    model.fit(X)
    print(f"epsilon ====>{eps}")
    plt.scatter(X[:,0],X[:,1], c=model.labels_)
    plt.show()


# In[612]:


X, y = make_moons(1000, noise=0.1)


# In[613]:


plt.scatter(X[:,0],X[:,1],c=y)


# In[614]:


model = KMeans(2)
model.fit(X)
plt.scatter(X[:,0],X[:,1],c=model.labels_)


# In[620]:


for eps in np.linspace(0.11,0.11241379310344828 ,10):
    model = DBSCAN(eps=eps)
    model.fit(X)
    print(f"eps ===> {eps} ")
    plt.scatter(X[:,0],X[:,1],c=model.labels_)
    plt.show()


# In[ ]:




