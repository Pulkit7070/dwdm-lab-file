import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("uci-ml-phishing-dataset.csv").drop(columns="id")
df["Result"] = df["Result"].map({1: 1, -1: 0})
print("Shape:", df.shape, "| classes:", df["Result"].value_counts().to_dict())

X, y = df.drop(columns="Result"), df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = RandomForestClassifier(n_estimators=200, random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC AUC :", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4))
print(classification_report(y_test, pred, target_names=["Phishing", "Legitimate"]))
print("Top 5 features:\n", pd.Series(model.feature_importances_, index=X.columns).nlargest(5).round(4))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Phishing", "Legitimate"])
plt.show()
