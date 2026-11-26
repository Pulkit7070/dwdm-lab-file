import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = (pd.read_csv("aioe_occupations.csv")
        .merge(pd.read_csv("occ_salary_2021.csv"), left_on="SOC Code", right_on="occ_code", how="left")
        .merge(pd.read_csv("occ_required_education.csv"), on="occ_code", how="left"))
cols = ["mean_salary_2021", "median_salary_2021", "occ_req_education"]
df[cols] = df[cols].fillna(df[cols].median())
df["Exposure"] = pd.qcut(df["AIOE"], 3, labels=["Low", "Medium", "High"])
print("Most exposed to AI:\n", df.nlargest(5, "AIOE")[["Occupation Title", "AIOE"]].to_string(index=False))
print("\nLeast exposed to AI:\n", df.nsmallest(5, "AIOE")[["Occupation Title", "AIOE"]].to_string(index=False))
print("\nCorrelation with AI exposure:\n", df[["AIOE"] + cols].corr()["AIOE"].round(2))

X, y = df[cols], df["Exposure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = DecisionTreeClassifier(max_depth=4, random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

df.plot.scatter(x="mean_salary_2021", y="AIOE", s=8, title="Salary vs AI exposure")
plt.xlabel("Mean salary 2021 (USD thousands)")
plt.show()
