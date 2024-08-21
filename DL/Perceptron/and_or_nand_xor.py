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
# %%
hx(*param)
#%%
param
# %% --------------------------------------------------- nand
df['nand'] = []
# %% --------------------------------------------------- xor
df['xor'] = []
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
