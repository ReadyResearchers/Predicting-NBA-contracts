# importing pandas
import pandas as pd
  
# merging two csv files
df = pd.concat(
    map(pd.read_csv, ['Salaries 2021:22.csv', 'PerGameAdvanced2022.csv']), ignore_index=True)

df_first_3= df.head(3)
print(df_first_3)