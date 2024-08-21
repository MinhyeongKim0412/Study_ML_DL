# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine

# %%
# 타이타닉 데이터 로드
df = pd.read_csv(r"C:\Users\kimmi\OneDrive\바탕 화면\KEPCO\KEPCO_class content\T.Lee_SpringBoot,Python\Python\Study_ML_DL-1\titanic.csv")

# %%
# 불필요한 컬럼 제거
df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'], inplace=True)

# %%
# 결측값 처리
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Embarked의 최빈값으로 대체

# %%
# 원-핫 인코딩 적용
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# %%
# 피처와 타겟 분리
X = df.drop(columns=['Survived'])
Y = df['Survived']

# %%
# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %%
# 스태킹 모델 구성
estimators = [
    ('svc', SVC(probability=True)),
    ('gnb', GaussianNB()),
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('lgbm', LGBMClassifier()),
    ('dt', DecisionTreeClassifier())
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# %%
# 스태킹 모델 학습
stack.fit(X_train, Y_train)

# %%
# 모델 평가
print(f"Stacking model score: {stack.score(X_test, Y_test)}")

# %%
# 테스트 데이터 준비
test = pd.DataFrame({
    'Pclass': [2],
    'Age': [26],
    'Sibsp': [2],
    'Parch': [0],
    'Fare': [X['Fare'].mean()],
    'Sex_male': [1],       # male인 경우 1
    'Embarked_Q': [0],     # Embarked가 Q가 아닌 경우 0
    'Embarked_S': [1]      # Embarked가 S인 경우 1
})

# %%
# 누락된 컬럼이 있는지 확인하고, 누락된 컬럼은 0으로 채워넣기
for col in X.columns:
    if col not in test.columns:
        test[col] = 0

# 컬럼 순서를 학습 데이터와 동일하게 맞추기
test = test[X.columns]

# %%
# 테스트 데이터 예측
test_prediction = stack.predict(test)
print(f"Test prediction: {test_prediction}")

# %%
# 모델 성능 평가
stack_score = stack.score(X_test, Y_test)
print(f"Stacking model test score: {stack_score}")

# %%
model = DecisionTreeClassifier()
model.get_params()
# %%
model = GaussianNB()
model.get_params()
# %%
model = LogisticRegression()
model.get_params()
# %%
model = RandomForestClassifier(criterion='entropy',n_estimators=200)
# %%
params = {'criterion':'gini'
        ,'n_estimators':300
        ,'n_jobs':-1
        ,'verbose':1}
model = RandomForestClassifier(**params)
model.get_params()
# %%
X,Y = load_wine().data, load_wine().target
# %%
def cross_val(estimator,x,y,cv=5):
    ind = np.arange(y.size)
    np.random.shuffle(ind)
    score=[]
    
    for cv in range(cv):
        tr_ind = ind[:int(y.size*0.8)]
        te_ind = ind[int(y.size*0.8)]
        X_train=X[tr_ind]
        X_test=X[te_ind]
        Y_train=Y[tr_ind]
        Y_test=Y[te_ind]
        
        estimator.fit(X_train,Y_train)
        score.append(estimator.score(X_test,Y_test))
        
# %%
X
# %%
Y
# %%
ind = np.arange(Y.size)
# %%
ind
# %%
np.random.shuffle(ind)
# %%
ind
# %%
np.unique(Y_train, return_counts=True)
# %%
pd.Series(Y_train).value_counts()
# %%
