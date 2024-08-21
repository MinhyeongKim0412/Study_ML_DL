#%%
import pandas as pd
# %%
exam_data = {'이름':['가영', '나영', '다영'],
            '수학':[90,80,70],
            '영어':[98,89,95],
            '음악':[85,95,100],
            '체육':[100,90,90]}
# %%
df = pd.DataFrame(exam_data)
# %%
df.set_index('이름', inplace=True)
print(df)
# %%
a = df.loc['가영', '음악']
# %%
print(a)
# %%
b=df.iloc[0,2]
# %%
print(b)
# %%
c=df.loc['가영', ['음악', '체육']]
print(c)
# %%
d=df.iloc[0,[2,3]]
print(d)
# %%
e=df.loc['가영', ['음악', '체육']]
print(e)
# %%
f=df.iloc[0,2:]
print(f)
# %%
