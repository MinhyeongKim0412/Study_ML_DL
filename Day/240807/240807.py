#%%
#수치적 미분
# limit x->0 ((f(x+h) - f(x))/h)
# MSE 사용하기

import numpy as np

#%%
X = np.random.randn(100)
Y = X*100 + 50
# %%
#X가 Y로 나오기까지의 가중치 W
#W에 대한 미분
W = np.random.randn(1)
# %%
Y_hat = X * W
# %%
#dW = np.mean(2*X*(Y_hat - Y))
#loss function에서 미분
#(Y_hat - Y)**2
for _ in range(10000):
    H = 1e-5
    fx = (X*W - Y)**2
    fxh = (X*(W+H) - Y)**2
    dW = np.mean((fxh - fx)/H)
    W -= dW * 1e-3
# %%
W
# -----------------------------------------------------------------------------


# %%
W = np.random.randn(1)
B = 0
# %%
Y_hat = X*W + B
# %%
def loss_MSE(Y, Y_hat):
    return np.mean((Y_hat - Y)**2)
# %%
loss_MSE(Y, Y_hat)
# %%
H = 1e-5

for _ in range(10000):
    Y_hat = X*W + B
    fx = loss_MSE(Y, Y_hat)

    Y_hat = X*(W+H) + B
    fxh = loss_MSE(Y, Y_hat)

    dW = (fxh - fx)/H

    fx = loss_MSE(Y, Y_hat)
    Y_hat = X*W + (H+B)

    fxh = loss_MSE(Y, Y_hat)
    dB = (fxh - fx)/H

    W -= dW*1e-3
    B -= dB*1e-3
# %%
W
# %%
B
# %%
X = np.random.randn(100,2)
Y = X @ np.array([[50],[1]])
# %%
W = np.random.randn(2,1)
# %%
H = 1e-3
#%%
print(W)
print("dw1 fx======")

Y_hat = X@W
fx = loss_MSE(Y, Y_hat)

W[0.0] += H
print(W)
print("dw1 fx======")
Y_hat = X@W
fxh = loss_MSE(Y, Y_hat)
dw1 = (fxh - fx)/H

print(W)
print("dw2 fx======")

W[1.0] += H
print(W)
print("dw2 fx======")
Y_hat = X@W
fxh = loss_MSE(Y, Y_hat)
dw2 = (fxh - fx)/H
# -------------------------------------------------------------------------
# %%
X = np.array([1,1],[1,0],[0,1],[0,0])

Y_and = np.array([],[],[],[])
Y_n_and = np.array([],[],[],[])
Y_or = np.array([0],[1],[1],[1])

W = np.random.randn(2,1)
W_and = W
W_n_and = W
W_or = W

B = 0
B_and = B
B_n_and = B
B_or = B
# %%
np.where(X@W + B > 0,5,1,0)
# %%
layer1_W = np.c_[W_or,W_n_and]
# %%
layer1_B = np.c_[B_or,B_n_and]
# --------------------------------------------
# %%
## 해석미분 / 수치미분
#-------------------------------------------------------------------------
#Supervised vs. Unsupervised vs. Reinforcement Learning
# %%
from sklearn.datasets import load_iris
import pandas as pd
# %%
iris = pd.DataFrame(load_iris()['data'],columns=load_iris()['feature_names'])
iris['species']=np.where(load_iris()['target']==0,'setosa',
                np.where(load_iris()['target']==1,'versicolor','virginica'))
# %%
iris
# %%
