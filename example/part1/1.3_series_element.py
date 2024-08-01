# %% [markdown]
# ### 시리즈 원소 선택

# %%
import pandas as pd

# %%
tup_data = ('김민형', '1998-04-12', '여', True)

# %%
sr = pd.Series(tup_data, index=['이름', '생년월일', '성별', '학생여부'])

# %%
print(sr)

# %% [markdown]
# ---

# %%
print(sr[0])
print(sr['이름'])

# %% [markdown]
# ---

# %%
# 여러 개 원소 고르기 - 리스트 활용

print(sr[[1, 2]])
print('\n')
print(sr[['생년월일', '성별']])

# %%
# 여러 개 원소 고르기 - 범위지정

print(sr[1:2])
print('\n')
print(sr[['생년월일', '성별']])
