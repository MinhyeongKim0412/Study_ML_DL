# %% [markdown]
# ### 딕셔너리 -> 시리즈 (변환)

# %%
import pandas as pd

# %%
dict_data = {'a': 1, 'b': 2, 'c': 3}

# %%
sr = pd.Series(dict_data)

# %%
print(type(sr))

# %%
print('\n')

# %%
print(sr)

# %%
