import pandas as pd

df = pd.read_csv('titanic.csv')
df_v2 = df[['Pclass', 'Sex', 'Age']].copy()
df_v2.to_csv('titanic_v2.csv', index=False)
print("Second Version Created")