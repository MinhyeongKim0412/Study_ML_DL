# %%
import pandas as pd

# %%
dict_data = {'c0': [1, 2, 3], 'c1': [4, 5, 6], 'c2': [7, 8, 9], 'c3': [10, 11, 12], 'c4': [13, 14, 15]}

# %%
df = pd.DataFrame(dict_data)

# %%
print(type(df))
print('\n')
print(df)

# %% [markdown]
# ---
# 
# ### 원소를 3개씩 담고있는 리스트를 5개 만들기, 그리고 각 리스트에 딕셔너리의 키 값은 총 5개(c0~c4)
