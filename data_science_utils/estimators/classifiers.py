from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=100000),
    "Random Forest": RandomForestClassifier(),
}

binary_classifiers = {
    "Logistic Regression": {"Estimator": LogisticRegression(), "Parameter Grid": {}},
    "Random Forest": {"Estimator": RandomForestClassifier(), "Parameter Grid": {}},
}

param_grid = {"Logistic Regression": {}}
