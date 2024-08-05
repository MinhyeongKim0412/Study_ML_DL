#%%
import pandas as pd

#%%
file_path = file_path = 'C:/Users/kimmi/OneDrive/바탕 화면/KEPCO/KEPCO_class content/T.Lee_SpringBoot,Python/Python/Study_ML_DL/example/part2/2.0_read_csv_sample.csv'

#%%
df1 = pd.read_csv(file_path)
print(df1)
print('\n')
# %%
df2 = pd.read_csv(file_path, header=None)
print(df2)
print('\n')
# %%
df3 = pd.read_csv(file_path, index_col=None)
print(df3)
print('\n')
# %%
df4 = pd.read_csv(file_path, index_col='c0')
print(df4)
print('\n')

# %%
