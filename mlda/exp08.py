import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

df = pd.read_csv("car_evaluation.csv", dtype=str)
levels = {"buying": ["low", "med", "high", "vhigh"], "maint": ["low", "med", "high", "vhigh"],
          "doors": ["2", "3", "4", "5more"], "persons": ["2", "4", "more"],
          "lug_boot": ["small", "med", "big"], "safety": ["low", "med", "high"]}
enc = OrdinalEncoder(categories=list(levels.values()))
X, y = enc.fit_transform(df[list(levels)]), df["class"]
print("Acceptability classes:", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)
model = DecisionTreeClassifier(random_state=21).fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

designs = [["low", "low", "4", "4", "big", "high"], ["high", "vhigh", "2", "4", "small", "med"],
           ["med", "med", "5more", "more", "med", "high"], ["vhigh", "low", "3", "2", "big", "high"]]
for d, p in zip(designs, model.predict(enc.transform(pd.DataFrame(designs, columns=list(levels))))):
    print(d, "->", p)

plt.figure(figsize=(14, 6))
plot_tree(model, max_depth=3, feature_names=list(levels), class_names=model.classes_, filled=True, fontsize=7)
plt.show()
