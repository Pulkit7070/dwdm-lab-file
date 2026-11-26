import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

rng = np.random.default_rng(21)
n = 500
df = pd.DataFrame({
    "Revenue_Growth": rng.uniform(2, 15, n), "ROI": rng.uniform(5, 20, n),
    "Cust_Satisfaction": rng.uniform(5, 10, n), "Retention_Rate": rng.uniform(60, 95, n),
    "Process_Efficiency": rng.uniform(60, 100, n), "Defect_Rate": rng.uniform(1, 10, n),
    "Engagement": rng.uniform(5, 10, n), "Training_Hours": rng.uniform(10, 50, n)})
df["Performance"] = (0.3 * df["Revenue_Growth"] + 0.2 * df["Cust_Satisfaction"]
                     + 0.2 * df["Process_Efficiency"] + 0.2 * df["Engagement"] + rng.normal(0, 2, n))
perspectives = {"Financial": ["Revenue_Growth", "ROI"], "Customer": ["Cust_Satisfaction", "Retention_Rate"],
                "Internal Process": ["Process_Efficiency", "Defect_Rate"],
                "Learning & Growth": ["Engagement", "Training_Hours"]}

X, y = df.drop(columns="Performance"), df["Performance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
model = RandomForestRegressor(n_estimators=300, random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("R2 :", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))

imp = pd.Series(model.feature_importances_, index=X.columns)
score = pd.Series({p: imp[c].sum() for p, c in perspectives.items()})
print("\nBalanced Scorecard weight of each perspective:\n", score.round(3))
score.plot(kind="bar", title="Balanced Scorecard Perspectives", rot=0)
plt.show()
