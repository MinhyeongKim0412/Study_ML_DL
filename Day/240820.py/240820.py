#%%
from itertools import combinations, product, zip_longest
from lightgbm import cv
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# %%
X, Y = load_wine().data, load_wine().target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
# %%
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)
# %%
plot_tree(tree_model)
# %%
## hyper_parameter
hyper_parameter = {"max_depth":3}
tree_model = DecisionTreeClassifier(**hyper_parameter)
# %%  정확도 확인
tree_model.fit(X_train,Y_train)
# %%
tree_model.score(X_test,Y_test)
# %%
model_pipe = Pipeline([
    ('pca', PCA(2)),
    ('scaler',StandardScaler()),
    ('svc',SVC())
])
# %%
sscaler = StandardScaler()
sscaler.fit(X_train)
X_train_std = sscaler.transform(X_train)

pca = PCA(2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train_std)

svc_model = SVC(probability=True)
svc_model.fit(X_train_pca,Y_train)
# %%
X_test_std = sscaler.transform(X_test)
X_test_pca = pca.transform(X_test_std)
svc_model.predict(X_test_pca)
# %%
svc_model = Pipeline([
    ('pca', PCA(3)),
    ('scaler',StandardScaler()),
    ('neighbor_estimator',KNeighborsClassifier())
])
# %%
svc_model.fit(X_train,Y_train)

###########################################################################

# kernel 'rbf, 'sigmoid', 'linear'
# C: 0.1 ~ 2 [20]
# %%
kernel = ['rbf', 'sigmoid', 'linear']
C = np.linspace(0.1, 2, 20)
scores = []
params = []

for c in C:
    for k in kernel:
        svc_model = SVC(kernel=k, C=c)
        svc_model.fit(X_train,Y_train)
        score = svc_model.score(X_test,Y_test)
        scores.append(score)
        params.append({'kernel':k,'C':c})
best_param = params[np.argmax(np.array(scores))]
best_score = np.max(np.array(scores))
# %%
svc_model = SVC(**best_param)
# %%
svc_model.fit(X_train,Y_train)
svc_model.score(X_test,Y_test)
# %%
scores[np.argmax(np.array(best_score))]
# %%
param_grid = {
    'C': np.linspace(0.1, 2, 20),
    'kernel': ['rbf', 'sigmoid', 'linear']
}
class GridSearchCrossVal:
    
    def __init__(self,estimator, param_grid, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_estimator_ = None
        self.best_params_ = {}
        self.best_score_ = 0
        
    def fit(self,x,y):
        param_list1 = list(self.param_grid.values())[0]
        param_list2 = list(self.param_grid.values())[1]
        param_list = list(product(param_list1, param_list2))
        score = []
        params = []
        
        for c,k in gsc.param_list:
            param = ({'C':c, 'kernel':k})
            params.append(param)
            self.estimator.set_params(**param)
            score = 0
            for i in range(self.cv):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
                self.estimator.fit(X,Y)
                score += self.estimator.score(X_test,Y_test)
            score = sum(score)/self.cv
            scores.append(score)
            
        self.best_score_ = np.max(scores)
        self.best_params_ = np.argmax(np.array(scores))
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)

        return self.best_score_
# %%
gsc  = GridSearchCrossVal(SVC(),[1,2,3],5)
# %%
param_grid
# %%
param_grid.keys()
# %%
X,Y = param_grid.items()
# %%
X[1]
# %%
params = []
for key in param_grid:
    for val in param_grid[key]:
        params.append({key:val})
# %%
params
# %%
(list(zip_longest(*param_grid.values(), fillvalue=['rbf','sigmoid','linear']))[3])
# %%
a = ['a','b']
b = [0.3, 0.2]
list(combinations([0.2, 'linear'], 2))
# %%