import pandas as pd

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("us_average_prices_item_summary.csv")
data = pd.concat([df, df.iloc[[3, 9, 15, 27, 41]]], ignore_index=True)
print("Rows with duplicates added:", len(data))
print(data[data.duplicated(keep=False)][["item", "first_price", "last_price"]])

clean = data.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(clean))
print("Removed:", len(data) - len(clean), "| remaining duplicates:", clean.duplicated().sum())

near = df.iloc[[19, 46]].copy()
near["item"] = near["item"].str.upper() + "  "
data2 = pd.concat([clean, near], ignore_index=True)
data2["key"] = data2["item"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
print("Exact duplicates found:", data2.duplicated().sum())
final = data2.drop_duplicates(subset=["key", "first_price", "last_price"]).drop(columns="key")
print("Rows after key based removal:", len(final))
final.to_csv("item_summary_dedup.csv", index=False)
