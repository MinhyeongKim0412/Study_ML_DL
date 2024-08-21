# %% [markdown]
# ### 시리즈 인덱스

# %%
import pandas as pd

# %%
list_data = ['2024-08-01', 3.14, 'ABC', 100, True]

# %%
sr = pd.Series(list_data)

# %%
print(sr)

# %% [markdown]
# ----------------------------------------------------------------------------------------------------------

# %%
idx = sr.index
val = sr.values

# %%
print(idx)
print('\n')
print(val)

# %%
