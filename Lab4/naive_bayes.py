import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# laoding the data 
data = pd.read_csv("2020_bn_nb_data.txt", sep="\t")

features = [
    "EC100",
    "EC160",
    "IT101",
    "IT161",
    "MA101",
    "PH100",
    "PH160",
    "HS101"
]

X = data[features]
y = data["QP"]


# calculating the CP 
def calculate_conditional_probabilities(X_train, y_train, features):

    probabilities = {}

    # All possible grades
    grades = [
        "AA", "AB", "BB", "BC",
        "CC", "CD", "DD", "F"
    ]

    # Number of possible grades
    K = len(grades)

    for feature in features:

        probabilities[feature] = {}

        for grade in grades:

            probabilities[feature][grade] = {}

            for qp_value in ["y", "n"]:

                # Number of students having:
                # feature = grade AND QP = qp_value
                count_grade_and_qp = (
                    (X_train[feature] == grade) &
                    (y_train == qp_value)
                ).sum()

                # Number of students having QP = qp_value
                count_qp = (y_train == qp_value).sum()

                # Laplace smoothing
                probability = (
                    count_grade_and_qp + 1
                ) / (
                    count_qp + K
                )

                probabilities[feature][grade][qp_value] = probability

    return probabilities


# testing on one stundent 
def predict_student(
    student,
    probabilities,
    p_y,
    p_n,
    features
):

    # Start with prior probabilities
    score_y = p_y
    score_n = p_n

    # Multiply P(grade | QP) for every course
    for feature in features:

        grade = student[feature]

        score_y *= probabilities[feature][grade]["y"]
        score_n *= probabilities[feature][grade]["n"]

    # Choose the class with the larger probability
    if score_y > score_n:
        prediction = "y"
    else:
        prediction = "n"

    return prediction, score_y, score_n


# running the classifier 20 times 
accuracies = []

print("Dataset shape:", data.shape)
print("\nRunning 20 experiments...\n")


for run in range(20):

    # ----------------------------------------------
    # 70% training / 30% testing
    # ----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=run
    )

    p_y = (y_train == "y").mean()
    p_n = (y_train == "n").mean()


    probabilities = calculate_conditional_probabilities(
        X_train,
        y_train,
        features
    )

    correct = 0

    for i in range(len(X_test)):

        student = X_test.iloc[i]

        prediction, score_y, score_n = predict_student(
            student,
            probabilities,
            p_y,
            p_n,
            features
        )

        if prediction == y_test.iloc[i]:
            correct += 1


    # Calculate accuracy

    accuracy = correct / len(X_test)

    accuracies.append(accuracy)

    print(
        f"Run {run + 1:2d}: "
        f"{correct}/{len(X_test)} correct "
        f"-> Accuracy = {accuracy:.4f}"
    )

print("\n" + "=" * 45)
print("FINAL RESULTS")
print("=" * 45)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
min_accuracy = np.min(accuracies)
max_accuracy = np.max(accuracies)

print(f"Mean accuracy : {mean_accuracy:.4f}")
print(f"Std deviation : {std_accuracy:.4f}")
print(f"Minimum       : {min_accuracy:.4f}")
print(f"Maximum       : {max_accuracy:.4f}")

print("\nMean accuracy (%):", mean_accuracy * 100)