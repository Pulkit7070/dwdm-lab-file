import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

X = StandardScaler().fit_transform(load_wine().data)
k = 3

km = KMeans(n_clusters=k, n_init=10, random_state=21).fit(X)
print("K-Means silhouette:", round(silhouette_score(X, km.labels_), 3))

D = cdist(X, X)
medoids = list(np.random.default_rng(21).choice(len(X), k, replace=False))
while True:
    labels = D[:, medoids].argmin(axis=1)
    new = [np.where(labels == j)[0][D[np.ix_(labels == j, labels == j)].sum(axis=1).argmin()] for j in range(k)]
    if sorted(new) == sorted(medoids):
        break
    medoids = new
print("K-Medoids medoids:", [int(m) for m in medoids], "silhouette:", round(silhouette_score(X, labels), 3))

Z = linkage(X, method="complete")
hc = fcluster(Z, t=k, criterion="maxclust")
print("Hierarchical silhouette:", round(silhouette_score(X, hc), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=km.labels_)
ax[0].set_title("K-Means clusters")
dendrogram(Z, truncate_mode="lastp", p=20, ax=ax[1])
ax[1].set_title("Dendrogram (complete linkage)")
plt.show()
