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
df.set_index('이름',inplace=True)
# %%
print(df)
# %%
print('\n')
# %%
df.iloc[0][3]=80
# %%
print(df)
print('\n')
# %%
df.loc['가영']['체육']=90
# %%
print(df)
print('\n')
# %%
df.loc['가영']['체육']=100
# %%
print(df)
# %%
df.loc['가영',['음악','체육']]=50
# %%
print(df)
print('\n')
# %%
df.loc['가영',['음악','체육']]= 100,50
# %%
print(df)
# %%
