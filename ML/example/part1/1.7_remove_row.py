# %%
import pandas as pd

# %%
exam_data = {'수학': [90, 90, 70], '영어': [9, 89, 95], '음악': [85, 95, 100], '체육': [100, 90, 90]}

# %%
df = pd.DataFrame(exam_data, index=['민형', '혜림', '신희'])

# %%
print(df)
print('\n')

# %%
df2 = df[:]
df2.drop('신희', inplace=True)

# %%
print(df2)
print('\n')

# %%
df3 = df[:]
df3.drop(['신희', '혜림'], axis=0, inplace=True)

# %%
print(df3)

# %%
# ---
import numpy as np

# %%
np.random.seed(100)
df = pd.DataFrame(np.random.choice([0, 1], (1000, 8)))

df.mean(0).argsort().head(1).index[0]
