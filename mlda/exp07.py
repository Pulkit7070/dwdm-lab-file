import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("tam_cloud_survey.csv")
constructs = ["PU", "PEOU", "PR", "US", "ITU"]

def cronbach_alpha(items):
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var().sum() / items.sum(axis=1).var())

for c in constructs:
    items = df.filter(regex=rf"^{c}\d$")
    df[c] = items.mean(axis=1)
    print(f"{c:<5} alpha = {cronbach_alpha(items):.3f}  mean = {df[c].mean():.2f}")

for formula in ["PU ~ PEOU", "ITU ~ PU + PEOU + PR + US"]:
    res = smf.ols(formula, data=df).fit()
    print(f"\n{formula}  (R2 = {res.rsquared:.3f})")
    print(res.params.drop("Intercept").round(3).to_string())

X, y = df[["PU", "PEOU", "PR", "US"]], (df["ITU"] >= df["ITU"].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = LogisticRegression().fit(X_train, y_train)
print("\nHigh intention prediction accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
new = pd.DataFrame([[4.6, 4.8, 2.8, 4.6], [3.2, 3.4, 4.6, 3.6], [4.0, 4.4, 3.8, 4.2]], columns=X.columns)
print("New respondents, P(high intention):", model.predict_proba(new)[:, 1].round(3))
