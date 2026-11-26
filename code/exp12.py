import numpy as np
import pandas as pd

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", usecols=[
    "EmployeeNumber", "Age", "Gender", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"])
rng = np.random.default_rng(21)
dirty = df.astype({"Age": float}).copy()
dirty.loc[rng.choice(len(df), 70, replace=False), "Age"] = np.nan
dirty.loc[rng.choice(len(df), 10, replace=False), "Age"] = 150
dirty.loc[rng.choice(len(df), 10, replace=False), "Gender"] = "M"
bad = rng.choice(len(df), 10, replace=False)
dirty.loc[bad, "YearsAtCompany"] = dirty.loc[bad, "TotalWorkingYears"] + 5
dirty = pd.concat([dirty, dirty.head(18)], ignore_index=True)

def quality(d):
    return pd.Series({
        "Completeness": 1 - d.isna().sum().sum() / d.size,
        "Uniqueness": d["EmployeeNumber"].nunique() / len(d),
        "Validity": (d["Age"].dropna().between(18, 65).mean() + d["Gender"].isin(["Male", "Female"]).mean()) / 2,
        "Consistency": (d["YearsAtCompany"] <= d["TotalWorkingYears"]).mean()})

clean = dirty.drop_duplicates().copy()
clean["Gender"] = clean["Gender"].replace({"M": "Male"})
clean.loc[~clean["Age"].between(18, 65), "Age"] = np.nan
clean["Age"] = clean["Age"].fillna(clean["Age"].median())
clean["YearsAtCompany"] = clean[["YearsAtCompany", "TotalWorkingYears"]].min(axis=1)
print(pd.DataFrame({"Dirty": quality(dirty), "Cleaned": quality(clean)}).round(4))
