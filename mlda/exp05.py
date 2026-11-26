import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

rng = np.random.default_rng(21)
n = 500
df = pd.DataFrame({"Page_Views": rng.integers(1, 50, n), "Ad_Clicks": rng.integers(0, 10, n),
                   "Time_Spent": rng.uniform(1, 20, n), "Bounce_Rate": rng.uniform(20, 90, n),
                   "Previous_Purchase": rng.integers(0, 2, n)})
score = (0.03 * df["Page_Views"] + 0.1 * df["Ad_Clicks"] + 0.05 * df["Time_Spent"]
         - 0.02 * df["Bounce_Rate"] + 0.3 * df["Previous_Purchase"] + rng.normal(0, 0.2, n))
df["Buy"] = (score > score.mean()).astype(int)

X, y = df.drop(columns="Buy"), df["Buy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = RandomForestClassifier(n_estimators=200, random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["No Purchase", "Purchase"]))

new = pd.DataFrame([[38, 9, 18.2, 25.0, 0], [8, 1, 3.4, 78.0, 1], [20, 3, 11.5, 60.0, 0]], columns=X.columns)
new["P(Purchase)"] = model.predict_proba(new)[:, 1].round(3)
print(new.to_string(index=False))

pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind="barh", title="Feature importance")
plt.show()
