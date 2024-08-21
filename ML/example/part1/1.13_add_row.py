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
print(df)
print('\n')
# %%
df.loc[3]=0 
print(df)
print('\n')
# %%
df.loc[4]=['라영', 90,80,70,60]
print(df)
# %%
df.loc['5행']=df.loc[3]
print(df)
# %%
