    
import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD AND SPLIT THE DATASET
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 2. GINI IMPURITY
# ==========================================

def gini_impurity(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return 1 - np.sum(probabilities ** 2)


# ==========================================
# 3. FIND THE BEST SPLIT
# ==========================================

def best_split(X, y, max_features):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    # Randomly select features for this node
    features = np.random.choice(
        X.columns.to_numpy(),
        size=min(max_features, len(X.columns)),
        replace=False
    )

    for feature in features:
        values = np.sort(X[feature].unique())

        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = (X[feature] <= threshold).to_numpy()
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            weighted_gini = (
                len(y_left) / len(y) * gini_impurity(y_left)
                + len(y_right) / len(y) * gini_impurity(y_right)
            )

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# ==========================================
# 4. BUILD ONE DECISION TREE
# ==========================================

def build_tree(X, y, depth=0, max_depth=5, max_features=5):
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]

    # Stop if pure or maximum depth reached
    if len(classes) == 1 or depth >= max_depth:
        return {"value": majority_class}

    # Stop if too few samples remain to split
    if len(y) < 2:
        return {"value": majority_class}

    feature, threshold = best_split(X, y, max_features)

    if feature is None:
        return {"value": majority_class}

    left_mask = (X[feature] <= threshold).to_numpy()
    right_mask = ~left_mask

    X_left = X.iloc[np.where(left_mask)[0]]
    X_right = X.iloc[np.where(right_mask)[0]]

    y_left = y[left_mask]
    y_right = y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return {"value": majority_class}

    left_subtree = build_tree(
        X_left, y_left, depth + 1, max_depth, max_features
    )

    right_subtree = build_tree(
        X_right, y_right, depth + 1, max_depth, max_features
    )

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }


# ==========================================
# 5. PREDICT USING ONE TREE
# ==========================================

def predict_one(sample, tree):
    if "value" in tree:
        return tree["value"]

    feature = tree["feature"]
    threshold = tree["threshold"]

    if sample[feature] <= threshold:
        return predict_one(sample, tree["left"])

    return predict_one(sample, tree["right"])


# ==========================================
# 6. RANDOM FOREST
# ==========================================

def train_random_forest(
    X,
    y,
    n_trees=10,
    max_depth=5,
    max_features=None,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    if max_features is None:
        max_features = max(1, int(np.sqrt(X.shape[1])))

    trees = []

    for i in range(n_trees):
        # Bootstrap sample: sample rows with replacement
        indices = rng.choice(
            len(X),
            size=len(X),
            replace=True
        )

        X_bootstrap = X.iloc[indices].reset_index(drop=True)
        y_bootstrap = y[indices]

        # Use a separate reproducible feature-selection stream
        np.random.seed(random_state + i)

        tree = build_tree(
            X_bootstrap,
            y_bootstrap,
            max_depth=max_depth,
            max_features=max_features
        )

        trees.append(tree)

        print(f"Trained tree {i + 1}/{n_trees}")

    return trees


# ==========================================
# 7. RANDOM FOREST PREDICTION
# ==========================================

def predict_forest(X, trees):
    all_predictions = []

    for _, sample in X.iterrows():
        tree_predictions = [
            predict_one(sample, tree)
            for tree in trees
        ]

        # Majority voting
        classes, counts = np.unique(
            tree_predictions,
            return_counts=True
        )

        prediction = classes[np.argmax(counts)]
        all_predictions.append(prediction)

    return np.array(all_predictions)


# ==========================================
# 8. EVALUATION
# ==========================================

def evaluate_model(y_true, y_pred):
    # Class 1 = benign (positive class)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


# ==========================================
# 9. TRAIN AND TEST THE RANDOM FOREST
# ==========================================

print("Training Random Forest from scratch...")

start_time = time.perf_counter()

forest = train_random_forest(
    X_train,
    y_train,
    n_trees=10,
    max_depth=5,
    max_features=5,
    random_state=42
)

training_time = time.perf_counter() - start_time

print("\nRandom Forest training complete.")
print(f"Training time: {training_time:.4f} seconds")

start_time = time.perf_counter()

y_pred = predict_forest(X_test, forest)

prediction_time = time.perf_counter() - start_time

print(f"Prediction time: {prediction_time:.4f} seconds")

print("\n--- Random Forest Evaluation ---")
evaluate_model(y_test, y_pred)
