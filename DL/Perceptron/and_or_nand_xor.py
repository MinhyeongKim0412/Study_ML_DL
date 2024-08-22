#%%
import numpy as np
import pandas as pd

from itertools import product
# %%
df = pd.DataFrame([[0,0],[0,1],[1,0],[1,1]])
# %%
df.columns = ['x1','x2']
# %% --------------------------------------------------- and
df['and'] = [0,0,0,1]
# %%
df
# %%
# if  x1*w1 + x2*w2 <= theta : 0
#     x1*w1 + x2*w2 > theta  : 1

def and_hx(w1,w2,theta):
    return np.where(df.x1*w1 + df.x2*w2 <= theta,0,1)
# %%
# w1,w2,theta
w1 = np.linspace(-1,1,20)
w2 = np.linspace(-1,1,20)
theta = np.linspace(-1,1,20)
params = product(w1,w2,theta)

for param in params:
    if np.all(and_hx(*param) == [0,0,0,1]):
        break
param
# %%
and_hx(*param)
#%%
def predict_and(x1,x2):
    return np.where(x1*param[0] + x2*param[1] <= param[2],0,1)
# %%
predict_and(1,1)
# %%
df
# %% --------------------------------------------------- or
df['or'] = [0,1,1,1]
# %%
df
#%%
param
# %%
## 미분 계산에서, 미분 대상은 무엇일까?
# 미분 대상은, 미분할 '함수'이다.
# Loss-function (mse, binary-crossentropy, categorical-crossentropy)
# KLD

#1. weight 가중치 초기화
x1 = df.x1
x2 = df.x2

y = df['and']

w1 = np.random.randn(1)
w2 = np.random.randn(1)
b = 0

#2. 가설함수
hx = x1*w1 + x2*w2 + b
loss = np.mean((hx-y)**2)
print(loss) # ========================

#3. 각 변수 node에 대한 미분값
dw1 = np.mean(x1*(hx-y))
dw2 = np.mean(x2*(hx-y))
db = np.mean(hx-y)

#4. 경사하강
w1 -= dw1*1e-3
w2 -= dw2*1e-3
b -= db*1e-3

hx = x1*w1 + x2*w2 + b
loss = np.mean((hx-y)**2)
print(loss) # ========================
# %%
hx = x1*w1 + x2*w2 + b
np.where(hx > 0,1,0)
# %%
loss
# %%
df
# %%
x1 = df.x1
x2 = df.x2

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
W= np.random.randn(3,1)
b = 0

hx = X@W + b
loss = np.mean((hx-y)**2)
# %%
# 1. 독립변수와 종속변수 분리
X = df.iloc[:, :-1]
Y = df.iloc[:,-1]

# 2. 가설함수
W = np.random.randn(X.shape[1], Y.shape[1])
b = np.zeros(Y.shape[1])
hx = lambda X,W,b : X@W + b

sigmoid = lambda X: 1/(1+np.exp(-X))
Y_hat = sigmoid(hx)

# 3. Loss function 정의
loss = lambda X: - np.mean(Y*np.log(X) + (1-Y)*np.log(1-X))

# 4. Loss function을 W, b에 대해 미분
learning_rate = 1e-3
epochs = 1000
h = 1e-5

for _ in range(epochs):
    
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
    fx = loss(Y_hat,Y)
    
    W[0,0] += h
    fxh = loss()
    dw1 = (fxh - fx)/h
    
    W[0,0] -= h
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
    fx = loss(Y_hat,Y)
    
    W[1,0] += h
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
    fx = loss(Y_hat,Y)
    fxh = loss(Y_hat,Y)
    b += h
    dw2 = (fxh - fx)/h
    W[1,0] -= h
    b += h
    
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
    fx = loss(Y_hat,Y)
    b += h
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
    fx = loss(Y_hat,Y)
    b -= h

# 5. 미분값을 대입
dW = np.array([dw1,dw2]).reshape(2,1)
W -= dW*learning_rate
b -= db*learning_rate

# Loss 계산
Y_hat = X@W + b
Y_hat = sigmoid(Y_hat)
fx = loss(Y_hat,Y)
# %%
h = 1e-5
# ----------------------------------------dw1
hx = X@W + b
Y_hat = 1/(1+np.exp(-hx))

fx = loss(Y_hat,Y)
fx
W[0,0] += h
hx = X@W + b
Y_hat = 1/(1+np.exp(-hx))
fxh = loss(Y_hat,Y)
dw1 = (fxh - fx)/h
dw1
W[0,0] -= h
# ----------------------------------------dw2
hx = X@W + b
Y_hat = 1/(1+np.exp(-hx))
fx = loss(Y_hat,Y)
W[1,0] += h
fxh = loss(Y_hat,Y)
dw2 = (fxh - fx)/h
W[1,0] -= h
# ----------------------------------------db
hx = X@W + b
Y_hat = 1/(1+np.exp(-hx))
fx = loss(Y_hat,Y)
b += h
hx = X@W + b
Y_hat = 1/(1+np.exp(-hx))
fxh = loss(Y_hat,Y)
db = (fxh - fx)/h
# %%
y*np.log()
# %%
dw1,dw2,db
# %%
learning_rate = 1e-3
dW = np.array([dw1,dw2]).reshape(2,1)
W -= dW * learning_rate
b -= db
# %%
Y_hat = X@W + b
Y_hat = sigmoid(Y_hat)
np.where(Y_hat > 0.5, 1.0)
# %%
def sigmoid(X):
    return 1/(1+np.exp(-X))
def loss(Y_hat,Y):
    return -np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))
def hx(W,b):
    Y_hat = X@W + b
    Y_hat = sigmoid(Y_hat)
# %%
X = df.iloc[:,:-1]
X = X.values
Y = df.iloc[:,-1]
Y = Y.values
Y = Y.reshape(-1,1)
X,Y
# %%
W = np.random.randn(X.shape[1], Y.shape[1])
b = np.zeros(Y.shape[1])
# %%
h = 1e-6
epochs = 1000
learning_rate = 1e-3
for i in range():
    rows = range(W.shape[0])
    cols = range(W.shape[1])
    for row in rows:
        for col in cols:
            dW = np.zeros_like(W)
            rows = range(W.shape[0])
            cols = range(W.shape[1])
    for col in cols:
        tmp = b[col]
        Y_hat = hx(W,b)
        fx = loss(Y_hat,Y)
##############################################################
#%%
from sklearn.datasets import load_breast_cancer
# %%
X = load_breast_cancer().data
Y = load_breast_cancer().target
# %%
Y
# %%
