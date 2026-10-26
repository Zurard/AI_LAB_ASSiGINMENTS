# doing this without the using library functions 

# dataset used : Wisconsin Breast Cancer dataset

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# loading the dataset : 
breast_cancer  = load_breast_cancer()
print(breast_cancer.target_names)

X = pd.DataFrame(
    data = breast_cancer.data , 
    columns= breast_cancer.feature_names)

Y = breast_cancer.target

# printing the first 5 rows 
print(X.head())

# printing feature shape and target shape 
print(X.shape)
print(Y.shape)

# printing features names and target names :  
print(X.columns.to_list())
print(breast_cancer.target_names)


# chekcing the class contribution 
classes, counts = np.unique(Y, return_counts=True)

print("\nClass distribution:")
for class_label, count in zip(classes, counts):
    print(f"{breast_cancer.target_names[class_label]}: {count}")


# 5. Check for missing values
print("\nMissing feature values:", X.isnull().sum().sum())


# 6. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# 7. Verify the split
print("\n--- Dataset Split ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

#Calculate Gini Impurity
def gini_impurity(y):
    if len(y) == 0:
        return 0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    gini = 1 - np.sum(probabilities ** 2)

    return gini


# Test the function
print("\n--- Gini Impurity Tests ---")

print("Entire dataset:", gini_impurity(Y))
print("Training dataset:", gini_impurity(y_train))
print("First 10 labels:", gini_impurity(y_train[:10]))

print("Pure group:", gini_impurity(np.array([0, 0, 0, 0])))
print("Mixed group:", gini_impurity(np.array([0, 0, 1, 1])))


# Find the best split
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    for feature in X.columns:
        values = np.sort(X[feature].unique())

        # Candidate thresholds: midpoints between consecutive values
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Weighted Gini impurity
            n = len(y)
            weighted_gini = (
                len(y_left) / n * gini_impurity(y_left)
                + len(y_right) / n * gini_impurity(y_right)
            )

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


# Test the function on the training dataset
feature, threshold, impurity = best_split(X_train, y_train)

print("\n--- Best Split ---")
print("Feature:", feature)
print("Threshold:", threshold)
print("Weighted Gini:", impurity)



# 10. Build the Decision Tree

def build_tree(X, y, depth=0, max_depth=5):
    # Find the majority class at the current node
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]

    # Stopping condition 1: all samples belong to one class
    if len(classes) == 1:
        return {"value": majority_class}

    # Stopping condition 2: maximum depth reached
    if depth >= max_depth:
        return {"value": majority_class}

    # Find the best split
    feature, threshold, impurity = best_split(X, y)

    # Stopping condition 3: no valid split found
    if feature is None:
        return {"value": majority_class}

    # Divide the dataset into left and right groups
    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold

    X_left = X[left_mask]
    X_right = X[right_mask]

    y_left = y[left_mask]
    y_right = y[right_mask]

    # Safety check: avoid empty child nodes
    if len(y_left) == 0 or len(y_right) == 0:
        return {"value": majority_class}

    # Recursively build the left and right subtrees
    left_subtree = build_tree(
        X_left, y_left, depth + 1, max_depth
    )

    right_subtree = build_tree(
        X_right, y_right, depth + 1, max_depth
    )

    # Return the current decision node
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }


# Train the tree
tree = build_tree(X_train, y_train, max_depth=5)

print("\n--- Decision Tree Built ---")
print(tree)



# 11. Predict the class for one sample

def predict_one(sample, tree):
    # If this is a leaf node, return its predicted class
    if "value" in tree:
        return tree["value"]

    # Get the feature and threshold for this decision node
    feature = tree["feature"]
    threshold = tree["threshold"]

    # Follow the appropriate branch
    if sample[feature] <= threshold:
        return predict_one(sample, tree["left"])
    else:
        return predict_one(sample, tree["right"])


# 12. Predict classes for multiple samples

def predict(X, tree):
    predictions = []

    for _, sample in X.iterrows():
        prediction = predict_one(sample, tree)
        predictions.append(prediction)

    return np.array(predictions)


# Test the trained tree
y_pred = predict(X_test, tree)

print("\n--- Prediction Results ---")
print("Actual labels:    ", y_test[:10])
print("Predicted labels: ", y_pred[:10])
print("Total predictions:", len(y_pred))



# 13. Evaluate the Decision Tree

def evaluate_model(y_true, y_pred):
    # Calculate True Positives, True Negatives,
    # False Positives, and False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate evaluation metrics
    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")


# Evaluate the predictions
evaluate_model(y_test, y_pred)
