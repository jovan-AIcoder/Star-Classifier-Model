# print the dataset info
print("The Star Dataset is downloaded at:")
print("https://www.kaggle.com/datasets/brsdincer/star-type-classification")
import pandas as pd
df = pd.read_csv("Stars.csv")
print(df.head())
print("\n")
print(df.describe())
print("\n")
print(df.info())
print("\n")
print(df.isnull().sum())