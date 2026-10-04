import pandas as pd

data = pd.read_csv(
    "data/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nLabel counts:")
print(data["label"].value_counts())