import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])
y = df.pop("Attrition").map({"Yes": 1, "No": 0})
X = StandardScaler().fit_transform(pd.get_dummies(df, drop_first=True).astype(float))
print("Encoded shape:", X.shape)

X_sel = SelectKBest(f_classif, k=10).fit_transform(X, y)
views = {
    "Feature Selection (top 10)": PCA(2).fit_transform(X_sel),
    "PCA": PCA(2).fit_transform(X),
    "t-SNE": TSNE(2, random_state=21).fit_transform(X),
    "UMAP": umap.UMAP(random_state=21).fit_transform(X),
}

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (name, Z) in zip(ax, views.items()):
    labels = KMeans(n_clusters=2, n_init=10, random_state=21).fit_predict(Z)
    print(f"{name:<28} K-Means silhouette = {silhouette_score(Z, labels):.3f}")
    a.scatter(Z[:, 0], Z[:, 1], c=labels, s=6)
    a.set_title(name)
plt.show()
