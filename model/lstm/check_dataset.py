import pandas as pd

train = pd.read_csv("tvtropes/train.balanced.csv")

print(train.describe())