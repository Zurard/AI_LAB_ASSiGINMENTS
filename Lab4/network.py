import pandas as pd
from pgmpy.causal_discovery import HillClimbSearch

from pgmpy.models import DiscreteBayesianNetwork

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("2020_bn_nb_data.txt", sep="\t")

courses = [
    "EC100", "EC160", "IT101", "IT161",
    "MA101", "PH100", "PH160", "HS101"
]

course_data = data[courses]

print("Data shape:", course_data.shape)

# Learn the Bayesian Network structure
hc = HillClimbSearch(
    scoring_method="bic-d",
    return_type="dag",
    show_progress=True
)

hc.fit(course_data)

learned_graph = hc.causal_graph_

print("\nLearned edges:")

for edge in learned_graph.edges():
    print(edge)
    
#creating baysian network 
model = DiscreteBayesianNetwork()

model.add_nodes_from(courses)
model.add_edges_from(learned_graph.edges())

# creating CPTs
model.fit(course_data)

# print("\n CPTs :")
# for cpd in model.get_cpds():
#     print("\n" , cpd)


# doing bayesian interference 

from pgmpy.inference import VariableElimination

# Create inference engine
inference = VariableElimination(model)

# Query PH100
result = inference.query(
    variables=["PH100"],
    evidence={
        "EC100": "DD",
        "IT101": "CC",
        "MA101": "CD"
    }
)

print("\nPH100 prediction:")
print(result)