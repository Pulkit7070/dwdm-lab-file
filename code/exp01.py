import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

prices = pd.read_csv("us_average_prices_monthly.csv")
items = pd.read_csv("us_average_prices_item_summary.csv")
print("Extracted:", prices.shape, items.shape)

df = prices[prices["category"].isin(["Fruit", "Vegetables", "Drinks"])].copy()
df["item"] = df["item"].str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["date", "price"]).drop_duplicates()
df = df[df["price"] > 0]
df["year"] = df["date"].dt.year
print("Transformed rows:", len(df))

con = sqlite3.connect("warehouse.db")
df.to_sql("fact_price", con, if_exists="replace", index=False)
print("Loaded rows:", con.execute("SELECT COUNT(*) FROM fact_price").fetchone()[0])

yearly = pd.read_sql("SELECT year, category, AVG(price) AS avg_price FROM fact_price GROUP BY year, category", con)
pivot = yearly.pivot(index="year", columns="category", values="avg_price").round(3)
print(pivot)
pivot.plot(marker="o", title="Yearly Average Price per Category")
plt.ylabel("Average price (USD)")
plt.show()
con.close()
