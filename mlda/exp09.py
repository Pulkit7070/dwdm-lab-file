import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

def fraud_model(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=21).fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, model.predict(X_test), target_names=["Normal", "Fraud"], digits=3))
    print("PR-AUC:", round(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]), 3))

cc = pd.read_csv("creditcard.csv.gz")
print("Credit card:", cc.shape, "| fraud rate:", round(cc["Class"].mean() * 100, 3), "%")
fraud_model(cc.drop(columns="Class"), cc["Class"], "Credit Card Transactions")

ps = pd.read_csv("paysim_sample_300k.csv")
ps = ps[ps["type"].isin(["TRANSFER", "CASH_OUT"])]
ps["is_transfer"] = (ps["type"] == "TRANSFER").astype(int)
cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "is_transfer"]
print("\nMobile payments (PaySim):", ps.shape, "| fraud rate:", round(ps["isFraud"].mean() * 100, 3), "%")
fraud_model(ps[cols], ps["isFraud"], "Mobile Payment Transactions")
