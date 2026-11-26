import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("nifty_500_stats_clean.csv")
cols = ["current_value", "book_value", "price_earnings", "dividend_yield", "roce", "roe", "sales_growth_3yr"]
df[cols] = df[cols].fillna(df[cols].median())
df["Cap"] = pd.cut(df["market_cap"], [0, 5000, 20000, float("inf")], labels=["Low Cap", "Medium Cap", "High Cap"])
print(df["Cap"].value_counts())

X = StandardScaler().fit_transform(df[cols].clip(df[cols].quantile(0.01), df[cols].quantile(0.99), axis=1))
y = df["Cap"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
clf = RandomForestClassifier(n_estimators=300, random_state=21).fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

Z = PCA(2).fit_transform(X)
clusters = KMeans(n_clusters=3, n_init=10, random_state=21).fit_predict(Z)
print("K-Means vs actual cap category (ARI):", round(adjusted_rand_score(y, clusters), 3))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(Z[:, 0], Z[:, 1], c=y.cat.codes, s=8)
ax[0].set_title("Actual cap category")
ax[1].scatter(Z[:, 0], Z[:, 1], c=clusters, s=8)
ax[1].set_title("K-Means clusters on PCA")
plt.show()
