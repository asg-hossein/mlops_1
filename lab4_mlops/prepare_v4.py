import pandas as pd

df = pd.read_csv('titanic_v3.csv')
df_v4 = pd.get_dummies(df, columns=['Sex'], prefix='Sex', drop_first=True)
df_v4.to_csv('titanic_v4.csv', index=False)
print(" 4rd Version Created (One-Hot Sex)")
