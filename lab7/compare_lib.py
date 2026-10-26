
import time
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and split the same dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Library models
models = {
    "Decision Tree - sklearn": DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    ),
    "Random Forest - sklearn": RandomForestClassifier(
        n_estimators=10,
        criterion="gini",
        max_depth=5,
        max_features=5,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label=1):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, pos_label=1):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, pos_label=1):.4f}")
    print(f"Training time:   {train_time:.4f} seconds")
    print(f"Prediction time: {predict_time:.4f} seconds")
