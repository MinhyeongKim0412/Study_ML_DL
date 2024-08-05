#%%
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
file_path = 'C:/Users/kimmi/OneDrive/바탕 화면/KEPCO/KEPCO_class content/T.Lee_SpringBoot,Python/Python/Study_ML_DL/example/part2/2.0.2_health_activity.csv'

#%%
df = pd.read_csv(file_path)

#%%
df_no_header = pd.read_csv(file_path, header=None)

#%%
print(df.head())
print(df_no_header.head())
#%%
df.fillna(0,inplace=True)

# %%
df.isna().sum()