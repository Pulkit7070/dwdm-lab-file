import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
features = ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
            "TrainingTimesLastYear", "TotalWorkingYears", "YearsAtCompany",
            "YearsWithCurrManager", "JobLevel", "MonthlyIncome"]
X, y = df[features], df["PerformanceRating"]
print("Performance ratings:", y.value_counts().sort_index().to_dict())
print("\nSalary hike decides the rating, so it is left out of the model:")
print(pd.crosstab(df["PercentSalaryHike"] >= 20, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("R2 :", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print(pd.DataFrame({"Actual": y_test.values[:8], "Predicted": pred[:8].round(2)}))

pd.Series(model.feature_importances_, index=features).sort_values().plot(kind="barh", title="What drives performance rating")
plt.show()
