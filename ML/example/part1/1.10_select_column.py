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
# %%
print(type(df))
print('\n')
# %%
math1 = df['수학']
print(math1)
print(type(math1))
print('\n')
# %%
english1 = df.영어
print(english1)
print(type(english1))
print('\n')
# %%
music_and_gym = df[['음악', '체육']]
print(music_and_gym)
print(type(music_and_gym))
print('\n')
# %%
math2 = df['수학']
print(math2)
print(type(math2))
print('\n')
# %%
