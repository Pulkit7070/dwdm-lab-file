import re
import pandas as pd

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

def clean(name):
    name = re.sub(r"\s+", " ", str(name).strip())
    name = re.sub(r"\s*,\s*", ", ", name)
    name = re.sub(r"\s*\(\s*", " (", name)
    name = re.sub(r"\s*\)", ")", name)
    return name.strip().title()

entries = ["  green tea", "Green    Tea", "GREEN TEA ", "green tea",
           "eggs , grade a", "EGGS,grade A ", "Eggs, Grade A",
           "rice ( basmati )", "Rice (Basmati) ", " RICE(basmati)"]
users = pd.DataFrame({"entered": entries})
users["clean"] = users["entered"].apply(clean)
print(users.assign(entered=users["entered"].map(repr)))
print("Distinct before:", users["entered"].nunique(), "| after:", users["clean"].nunique())

df = pd.read_csv("us_average_prices_item_summary.csv")
df["item_clean"] = df["item"].apply(clean)
print("Item names changed:", (df["item"] != df["item_clean"]).sum())
df.to_csv("item_summary_cleaned.csv", index=False)
print(df[["item", "item_clean"]].head(10))
