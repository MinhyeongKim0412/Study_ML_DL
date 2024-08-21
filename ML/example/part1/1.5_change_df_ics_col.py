# %%
import pandas as pd

# %%
df = pd.DataFrame([[27, '여', '광주'], [27, '남', '인천']], index=['민형', '제모'], columns=['나이', '성별', '지역'])

# %%
print(df)
print('\n')
print(df.index)
print('\n')
print(df.columns)
