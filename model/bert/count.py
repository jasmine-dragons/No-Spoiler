import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(len(train) + len(test))

df = pd.concat([train, test])

df_spoiler = df[df['spoiler'] == True]
print(len(df_spoiler) / len(df))