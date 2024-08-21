#%%
import pandas as pd

#%%
file_path = 'C:/Users/kimmi/OneDrive/바탕 화면/KEPCO/KEPCO_class content/T.Lee_SpringBoot,Python/Python/Study_ML_DL/example/part2/2.0.3_read_json_sample.json'

#%%
df = pd.read_json(file_path)

#%%
print(df)

# %%
print(df.index)
# %%
