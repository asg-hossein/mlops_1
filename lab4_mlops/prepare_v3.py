import pandas as pd

df = pd.read_csv('titanic_v2.csv')
df_v3 = df.copy()
df_v3['Age'] = df_v3['Age'].fillna(df_v3['Age'].mean())
df_v3.to_csv('titanic_v3.csv', index=False)
print("3rd Version Created (Age)")
