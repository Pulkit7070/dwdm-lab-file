import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

euclidean = lambda a, b: np.sqrt(((a - b) ** 2).sum())
manhattan = lambda a, b: np.abs(a - b).sum()

p, q = np.array([5, 12, 30, 7]), np.array([9, 4, 25, 11])
print("Euclidean(p, q):", round(euclidean(p, q), 4))
print("Manhattan(p, q):", manhattan(p, q))

df = pd.read_csv("nifty_500_stats_clean.csv")
cols = ["price_earnings", "roe", "roce", "dividend_yield", "sales_growth_3yr"]
df = df[df["category"] == "Nifty 50"].dropna(subset=cols)
df = df[df["price_earnings"] < 150].set_index("symbol")
X = pd.DataFrame(MinMaxScaler().fit_transform(df[cols]), index=df.index)

names = ["RELIANCE", "ONGC", "IOC", "BPCL", "COALINDIA", "NTPC"]
E = pd.DataFrame([[euclidean(X.loc[a], X.loc[b]) for b in names] for a in names], names, names)
M = pd.DataFrame([[manhattan(X.loc[a], X.loc[b]) for b in names] for a in names], names, names)
print("\nEuclidean distance matrix\n", E.round(3))
print("\nManhattan distance matrix\n", M.round(3))
