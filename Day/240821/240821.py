#%% ------------------------------------------------------------------------------------------ Library import
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% ------------------------------------------------------------------------------------------ def X,Y
X = datasets.load_diabetes().data
Y = datasets.load_diabetes().target
# %% ------------------------------------------------------------------------------------------ StandardScaler
sscaler = StandardScaler()
sscaler.fit(X)
X_std = sscaler.transform(X)
# %%
X_train, X_test, Y_train, Y_test = train_test_split(X_std,Y)
# %% ------------------------------------------------------------------------------------------ LinearRegression model
model = LinearRegression()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# %% ------------------------------------------------------------------------------------------ check Predict
model.predict(X_test)
X_test @ model.coef_.reshape(10,1) + model.intercept_
# %% ------------------------------------------------------------------------------------------ check Parameter
model.get_params()
# %% ------------------------------------------------------------------------------------------ DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# %% ------------------------------------------------------------------------------------------ SVC
model = SVR()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# %% ------------------------------------------------------------------------------------------ RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# %%
model.feature_importances_
# %%
feature_names = datasets.load_breast_cancer().feature_names
# %% ------------------------------------------------------------------------------------------ def col
X_axis = feature_names[np.argsort(model.feature_importances_)[::-1]]
# %%
Y_axis = model.feature_importances_[np.argsort(model.feature_importances_)[::-1]]
# %%
plt.bar(X_axis,Y_axis)
# %%
X_axis[5]
# %%
datasets.load_breast_cancer().feature_names[np.argsort(col)[::-1]][:5]
# %%
X_axis[:5]
# %%
model = LinearRegression()
model.fit(X_train,Y_train)
# %%
model.coef_
# %% ------------------------------------------------------------------------------------------ 계수를 positive로 변환
coef = np.where(model.coef_<0, model.coef_*(-1), model.coef_)
# %% ------------------------------------------------------------------------------------------ 데이터의 컬럼 이름
col = datasets.load_breast_cancer().feature_names
# %% ------------------------------------------------------------------------------------------ 계수 크기 순서 설정
ind = np.argsort(coef)[::-1]
# %% ------------------------------------------------------------------------------------------ x축:크기순서의 컬럼명, y축:계수 크기순정렬
X_axis = col[ind]
Y_axis = coef[ind]
# %% ------------------------------------------------------------------------------------------ 큰 순대로 5개 막대그래프 시각화
plt.bar(X_axis[:5], Y_axis[:5])
# %% ------------------------------------------------------------------------------------------ test
# cross_var_score 구하기
# 랜덤하게 5번 샘플링, 각 f1 score의 평균 구하기
# 각 모델들의 f1 score가 높은 순으로 모델들의 막대그래프 출력하기

cross_val_score(model, X, Y, scoring='accuracy')
# %%
#%%
X = load_digits().data
Y = load_digits().target

#%%
def cross_f1(model, x, Y, cv=5):
    f1 = 0
    for _ in range(cv):
        X_train, X_test, Y_train, Y_test = train_test_split(x, Y)
        model.fit(X_train, Y_train)
        model.score(X_test, Y_test)
        pred = model.predict(X_test)
        f1 += f1_score(Y_test, pred, average='macro')
    return f1 / cv

#%%
class ModelImportance:
    
    def __init__(self, *models):
        self.clfs = [model for model in models]
        self.clf_f1 = {}
        
    def cross_f1(self, X, Y, cv=5):
        for model in self.clfs:
            f1 = 0
            for _ in range(cv):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
                model.fit(X_train, Y_train)
                pred = model.predict(X_test)
                f1 += f1_score(Y_test, pred, average='macro')
            self.clf_f1[repr(model)[:repr(model).find('(')]] = f1 / cv
        return self.clf_f1
            
    def bar_plot(self, X, Y):
        self.cross_f1(X, Y)
        
    def cross_val(self, X, Y, scoring, cv=5):
        scoring_dic = {
            "accuracy":accuracy_score,
            "recall":recall_score,
            "precision":precision_score,
            "f1_score":f1_score
        }
        for model in self.clfs:
            f1 = 0
            for _ in range(cv):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
                model.fit(X_train, Y_train)
                pred = model.predict(X_test)
                f1 += scoring_dic(Y_test, pred, average='macro')
            self.clf_f1[repr(model)[:repr(model).find('(')]] = f1 / cv
        return self.clf_f1
    
    def display(self,X,Y):
        

#%%
aa = ModelImportance(DecisionTreeClassifier(), RandomForestClassifier())

#%%
aa.cross_f1(X, Y)
aa.cross_f1
# %%
plt.bar(aa.clf_f1.keys(), aa.clf_f1.values())
# %%
scoring_dic = {
    "accuracy":accuracy_score,
    "recall":recall_score,
    "precision":precision_score,
    "f1_score":f1_score
}
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
# %%
param = {'average':'macro'}
scoring_dic[](Y_test, model.predict(X_test),**param)
# %%
