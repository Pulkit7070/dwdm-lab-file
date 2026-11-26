import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

data, meta = arff.loadarff("iris.arff")
df = pd.DataFrame(data)
df["class"] = df["class"].str.decode("utf-8")
colors = df["class"].astype("category").cat.codes

df.hist(figsize=(8, 5))
plt.suptitle("Histogram of every attribute")
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, :4], c=colors, figsize=(8, 8))
plt.suptitle("Scatter plot matrix of iris (colour = class)")
plt.show()

plt.scatter(df["sepallength"], df["petallength"], c=colors)
plt.xlabel("sepallength")
plt.ylabel("petallength")
plt.title("sepallength vs petallength")
plt.show()
