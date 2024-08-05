#%%
import pandas as pd
# %%
exam_data = {'수학': [90,90,70], '영어': [98,89,95],
            '음악': [85,95,100], '체육': [100,90,90]}
# %%
df = pd.DataFrame(exam_data, index=['가영', '나영', '다영'])
print(df)
print('\n')
# %%
label1=df.loc['가영']
# %%
position1 = df.iloc[0]
# %%
print(label1)
print('\n')
print(position1)
# %%
label2 = df.loc[['가영', '나영']]
# %%
position2 = df.iloc[[0,1]]
# %%
print(label2)
print('\n')
print(position2)
# %%
label3 = df.loc[['가영', '다영']]
# %%
position3 = df.iloc[0:1]
# %%
print(label3)
print('\n')
print(position3)
# %%
