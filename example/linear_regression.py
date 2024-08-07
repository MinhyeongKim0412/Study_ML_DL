# -------------------------------- 선형회귀 ----------------------------
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
TV = np.random.randint(1,10,1000)

# %%
price = np.random.randint(1,500,1)
# %%
sales = TV * price + np.random.normal(5,25,1000)
# %%
plt.scatter(TV,sales)
# %%
#TV*price
price_hat = np.random.normal(1,3,1)
# %%
sales_hat = TV * price_hat
# %%
#sales_hat과 sales의 오차 계산

def loss_fn(y,y_hat):
    return np.mean((y_hat - y)**2)
loss_fn(sales,sales_hat)

# %%
err=[]
price_hat_range = np.linspace(0,10,100)
for price_hat in price_hat_range:
    sales_hat = TV * price_hat
    err.append(loss_fn(sales,sales_hat))

# %%
plt.plot(price_hat_range,err)
# %%
price_hat = 200
sales_hat = TV * price_hat
loss_fn(sales,sales_hat)
# %%
y_hat = TV * price_hat
np.mean((y_hat- y)**2)
# %%
price_hat = np.random.randn(1)
# %%
for _ in range(10000):
    sales_hat = TV*price_hat
    dprice_hat = np.mean(2*TV*(sales_hat-sales))
    price_hat -= dprice_hat*1e-3
# %%
loss_fn(sales,sales_hat)








# %%
price = 1000.
bias = -300.
# %%
sales = TV * price + bias + np.random.normal(200,100,1000)
# %%
plt.scatter(TV,sales)
# %%
price_hat = np.random.randn(1)
bias_hat = 200.
# %%
for i in range(10000):
    sales_hat = TV * price + bias_hat
    dprice_hat = np.mean(2*TV*(sales_hat - sales))
    dbias_hat = np.mean(2*(sales_hat - sales))
    price_hat -= dprice_hat * 1e-3
    bias_hat -= dbias_hat * 1e-3
# %%
price_hat
# %%
bias_hat








# %%
TV = np.random.randint(1,10,5000)
Radio = np.random.randint(3,15,5000)
#%%
tv_price_hat = np.random.randn(1)
radio_price_hat = np.random.randn(1)
bias_hat = 0.
#%%
for _ in range(10000):
    sales_hat = TV*tv_price_hat + Radio*radio_price_hat + bias_hat
    dtv_price_hat = np.mean(2*TV*(sales_hat - sales))
    dradio_price_hat = np.mean(2*Radio*(sales_hat - sales))
    dbias_hat = np.mean(sales_hat - sales)

    tv_price_hat -= dtv_price_hat*1e-3
    radio_price_hat -= dradio_price_hat*1e-3
    bias_hat -= dbias_hat*1e-3
# %%
tv_price_hat
# %%
radio_price_hat
# %%
bias_hat








# %%
