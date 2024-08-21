#%%
import pandas as pd
# %%
exam_data = {'수학': [90,90,70], '영어': [98,89,95],
            '음악': [85,95,100], '체육': [100,90,90]}
# %%
df = pd.DataFrame(exam_data, index=['가영', '나영', '다영'])
# %%
print(df)
print('\n')
# %%
df4=df[:]
# %%
df4.drop('수학', axis=1, inplace=True)
# %%
print(df4)
# %%
print('\n')
# %%
