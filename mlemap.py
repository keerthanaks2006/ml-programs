
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mle = LogisticRegression(penalty=None, max_iter=5000)
mle.fit(X_train, y_train)

map_l2 = LogisticRegression(penalty='l2', C=1.0, max_iter=5000)
map_l2.fit(X_train, y_train)

map_l1 = LogisticRegression(
    penalty='l1', solver='liblinear', C=1.0, max_iter=5000
)
map_l1.fit(X_train, y_train)

models = {
    "MLE (No Regularization)": mle,
    "MAP-L2 (Ridge)": map_l2,
    "MAP-L1 (Lasso)": map_l1
}

for name, model in models.items():
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("Coefficients:", model.coef_[0])

print("\nParameter Comparison")
print("MLE coefficients :", mle.coef_[0])
print("L2 coefficients  :", map_l2.coef_[0])
print("L1 coefficients  :", map_l1.coef_[0])

print("\nNumber of zero coefficients:")
print("MLE:", np.sum(np.isclose(mle.coef_[0], 0, atol=1e-6)))
print("L2 :", np.sum(np.isclose(map_l2.coef_[0], 0, atol=1e-6)))
print("L1 :", np.sum(np.isclose(map_l1.coef_[0], 0, atol=1e-6)))
