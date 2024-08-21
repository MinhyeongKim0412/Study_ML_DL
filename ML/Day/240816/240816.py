# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd
import numpy as np
# %%
def make_df(x):
    df = x.data
    df = pd.DataFrame(df)
    df.columns = x.feature_names
    df['target'] = x.target
    return df

# %%
iris = make_df(load_iris())

# %%
x = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y)

# %%
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

# %%
np.argmax(tree_model.predict_proba(x_test), axis=1)

# %%
np.argmax(tree_model.predict(x_test))

# %%
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# %%
np.argmax(lr_model.predict_proba(x_test), axis=1)

# %%
x_train.shape

# %%
lr_model = LogisticRegression()
bayes_model = GaussianNB()
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier()

# %%
lr_model.fit(x_train, y_train)
bayes_model.fit(x_train, y_train)
svm_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)

# 주석 처리된 동일한 데이터셋 분할을 반복하지 않음
# x_train, x_test, y_train, y_test = train_test_split(x, y)

# %%
lr_pred = lr_model.predict_proba(x_test)
bayes_pred = bayes_model.predict_proba(x_test)
svm_pred = svm_model.predict_proba(x_test)
rf_pred = rf_model.predict_proba(x_test)

# %%
lr_pred

# %%
lr_pred.shape

# %%
pred = lr_pred + bayes_pred + svm_pred + rf_pred

# %%
accuracy_score(y_test, np.argmax(pred, axis=1))

# %%
np.argmax(pred, axis=1)

# %%
# 수정된 VoteSoft 클래스
class VoteSoft:
    def __init__(self, *args):
        self.clfs = [arg for arg in args]
        self.clf_fit = {}

    def fit(self, x, y):
        for idx, clf in enumerate(self.clfs):
            self.clf_fit['clf' + str(idx + 1)] = clf.fit(x, y)

    def predict(self, x):
        pred = np.zeros((x.shape[0], len(self.clfs)))
        for idx, clf in enumerate(self.clf_fit.values()):
            pred[:, idx] = np.argmax(clf.predict_proba(x), axis=1)
        self.pred_result = np.apply_along_axis(self.vote, 1, pred)
        return self.pred_result

    def score(self, x, y):
        return np.sum(self.predict(x) == y) / x.shape[0]

    def f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='weighted')

    def vote(self, x):
        dt, cnt = np.unique(x, return_counts=True)
        return dt[np.argmax(cnt)]
    
    def confusion_matrix(self, x, y):
        pred = self.predict(x)
        return confusion_matrix(y, pred)

# %%
vote = VoteSoft(LogisticRegression(), RandomForestClassifier())
# %%
vote.fit(x_train, y_train)
# %%
vote.predict(x_test).shape

# %%
# 올바르게 접근할 수 있도록 수정
vote.clf_fit['clf1'].predict(x_test)
# %%
vote.f1_score(x_test, y_test)

# %%
vote.confusion_matrix(x_test, y_test)
# %%
pred = np.argmax(vote.predict(x_test),1)
# %%
dic = {}
# %%
dic.get(i,0)
# %%
dic = {}
for i in range(3):
    for j in range(len(pred)):
        if pred[j] == i:
            dic.get(i,0) += (pred[j]==y_test[j])
# %%
