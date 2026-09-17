import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from pgmpy.causal_discovery import HillClimbSearch


# ==========================================================
# 1. LOAD DATA
# ==========================================================

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

target = "QP"

columns = features + [target]

# Possible states for course grades
course_grades = [
    "AA", "AB", "BB", "BC",
    "CC", "CD", "DD", "F"
]

# Possible states for QP
qp_states = ["y", "n"]

# State space of every variable
states = {}

for feature in features:
    states[feature] = course_grades

states[target] = qp_states


print("Dataset shape:", data.shape)


# ==========================================================
# 2. CALCULATE CPTs
# ==========================================================

def calculate_cpts(train_data, model):

    cpts = {}

    # NetworkX topological ordering
    nodes = list(__import__("networkx").topological_sort(model))

    for node in nodes:

        parents = list(model.predecessors(node))

        cpts[node] = {
            "parents": parents,
            "probabilities": {}
        }

        node_states = states[node]

        # --------------------------------------------------
        # NODE WITHOUT PARENTS
        # P(node)
        # --------------------------------------------------

        if len(parents) == 0:

            total = len(train_data)

            for node_state in node_states:

                count = (
                    train_data[node] == node_state
                ).sum()

                # Laplace smoothing
                probability = (
                    count + 1
                ) / (
                    total + len(node_states)
                )

                cpts[node]["probabilities"][
                    (node_state,)
                ] = probability


        # --------------------------------------------------
        # NODE WITH PARENTS
        # P(node | parents)
        # --------------------------------------------------

        else:

            parent_state_lists = [
                states[parent]
                for parent in parents
            ]

            parent_combinations = itertools.product(
                *parent_state_lists
            )

            for parent_values in parent_combinations:

                subset = train_data

                # Select rows matching parent values
                for parent, value in zip(
                    parents,
                    parent_values
                ):

                    subset = subset[
                        subset[parent] == value
                    ]

                total = len(subset)

                for node_state in node_states:

                    count = (
                        subset[node] == node_state
                    ).sum()

                    # Laplace smoothing
                    probability = (
                        count + 1
                    ) / (
                        total + len(node_states)
                    )

                    key = parent_values + (node_state,)

                    cpts[node]["probabilities"][key] = probability

    return cpts


# ==========================================================
# 3. CALCULATE JOINT PROBABILITY
# ==========================================================

def calculate_joint_probability(
    student,
    model,
    cpts
):

    probability = 1.0

    # NetworkX topological ordering
    nodes = list(__import__("networkx").topological_sort(model))

    for node in nodes:

        parents = cpts[node]["parents"]

        node_state = student[node]

        # No parents
        if len(parents) == 0:

            key = (node_state,)

        # Has parents
        else:

            parent_values = tuple(
                student[parent]
                for parent in parents
            )

            key = parent_values + (node_state,)

        probability *= cpts[node]["probabilities"][key]

    return probability


# ==========================================================
# 4. PREDICT QP
# ==========================================================

def predict_qp(
    student,
    model,
    cpts
):

    # ------------------------------------------------------
    # Case 1: QP = y
    # ------------------------------------------------------

    student_y = student.copy()

    student_y["QP"] = "y"

    probability_y = calculate_joint_probability(
        student_y,
        model,
        cpts
    )


    # ------------------------------------------------------
    # Case 2: QP = n
    # ------------------------------------------------------

    student_n = student.copy()

    student_n["QP"] = "n"

    probability_n = calculate_joint_probability(
        student_n,
        model,
        cpts
    )


    # ------------------------------------------------------
    # Choose larger probability
    # ------------------------------------------------------

    if probability_y > probability_n:
        prediction = "y"
    else:
        prediction = "n"

    return prediction, probability_y, probability_n


# ==========================================================
# 5. RUN 20 EXPERIMENTS
# ==========================================================

accuracies = []

print("\nRunning 20 Bayesian Network experiments...\n")


for run in range(20):

    print("=" * 50)
    print(f"Run {run + 1}")
    print("=" * 50)


    # ------------------------------------------------------
    # 70% TRAINING / 30% TESTING
    # ------------------------------------------------------

    train_data, test_data = train_test_split(
        data[columns],
        test_size=0.30,
        random_state=run
    )

    print(
        f"Training samples: {len(train_data)}, "
        f"Testing samples: {len(test_data)}"
    )


    # ------------------------------------------------------
    # 6. LEARN STRUCTURE
    # ------------------------------------------------------

    hc = HillClimbSearch(
        scoring_method="bic-d",
        return_type="dag",
        max_iter=10000,
        show_progress=False
    )

    hc.fit(train_data)

    learned_graph = hc.causal_graph_


    # ------------------------------------------------------
    # 7. CALCULATE CPTs
    # ------------------------------------------------------

    cpts = calculate_cpts(
        train_data,
        learned_graph
    )


    # ------------------------------------------------------
    # 8. PREDICT TEST STUDENTS
    # ------------------------------------------------------

    correct = 0

    for _, student in test_data.iterrows():

        prediction, probability_y, probability_n = predict_qp(
            student,
            learned_graph,
            cpts
        )

        actual = student["QP"]

        if prediction == actual:
            correct += 1


    # ------------------------------------------------------
    # 9. CALCULATE ACCURACY
    # ------------------------------------------------------

    accuracy = correct / len(test_data)

    accuracies.append(accuracy)


    # ------------------------------------------------------
    # 10. PRINT LEARNED NETWORK
    # ------------------------------------------------------

    print("\nLearned edges:")

    for edge in learned_graph.edges():
        print(edge)

    print(
        f"\nCorrect: {correct}/{len(test_data)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================================
# 11. FINAL RESULTS
# ==========================================================

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
min_accuracy = np.min(accuracies)
max_accuracy = np.max(accuracies)

print(f"Mean accuracy : {mean_accuracy:.4f}")
print(f"Std deviation : {std_accuracy:.4f}")
print(f"Minimum       : {min_accuracy:.4f}")
print(f"Maximum       : {max_accuracy:.4f}")

print(
    f"\nMean accuracy (%): "
    f"{mean_accuracy * 100:.2f}%"
)