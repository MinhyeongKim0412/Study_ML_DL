#%%
import pandas as pd

#%%
file_path = r'C:\Users\kimmi\OneDrive\바탕 화면\KEPCO\KEPCO_class content\T.Lee_SpringBoot,Python\Python_pjt_kmh\health.activity.day_summary.csv.csv'

#%%
df1 = pd.read_csv(file_path)
# %%
df2 = pd.read_csv(file_path, header=None)
# %%
print(df1)
# %%
print(df2)
# %%
