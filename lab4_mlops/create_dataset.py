import pandas as pd
from catboost.datasets import titanic

train, test = titanic()
df = pd.concat([train, test], ignore_index=True)

df.to_csv('titanic.csv', index=False)
print("First Version Created", df.shape)