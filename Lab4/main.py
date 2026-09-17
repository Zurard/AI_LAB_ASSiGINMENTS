# what is a bayesian network  /? 
    # ans : a graph showing probablisitic relationship between  variable + probabilites describing those relationships 
    
# What is CPT ?? 
    # ans : CPT stands for Conditional Probability Table 
        # suppose a ----> b then to fo find P(b) is same as P(b | a) we need to consider a as b depends on so we create a table to store all the values for this  
        
# learn dependencies means since u have teh data find the relationship between each coloumn
    # in this we have 2 types of learing 
    #  1.) structure :  tell us which varaibles are connected 
    #  2.) parameter :  given those connections what is the probability 
    
# what is naive bayes ?? /
# is is the first classifier i need to build in our case if in know if the kid is qualified or not then other varaibles 




import subprocess
import sys


def print_header(title):
    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def run_file(filename):
    print_header(f"RUNNING {filename}")

    result = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("\n[Warnings / Errors]")
        print(result.stderr)

    if result.returncode != 0:
        print(f"\n{filename} failed with exit code {result.returncode}")
        return False

    return True


def main():

    print("=" * 75)
    print("AI LAB - BAYESIAN NETWORK AND NAIVE BAYES")
    print("=" * 75)

    print("\nDataset: 2020_bn_nb_data.txt")
    print("Experiments:")
    print("1. Bayesian Network structure learning and inference")
    print("2. Naive Bayes classification")
    print("3. Bayesian Network classification with dependencies")

    # -------------------------------------------------------
    # 1. Bayesian Network
    # -------------------------------------------------------

    print_header(
        "PART 1 - BAYESIAN NETWORK STRUCTURE AND INFERENCE"
    )

    if not run_file("network.py"):
        print("\nStopping because network.py failed.")
        return

    # -------------------------------------------------------
    # 2. Naive Bayes
    # -------------------------------------------------------

    print_header(
        "PART 2 - NAIVE BAYES CLASSIFICATION"
    )

    if not run_file("naive_bayes.py"):
        print("\nStopping because naive_bayes.py failed.")
        return

    # -------------------------------------------------------
    # 3. Dependent Bayesian Network
    # -------------------------------------------------------

    print_header(
        "PART 3 - BAYESIAN NETWORK CLASSIFICATION"
    )

    if not run_file("dependent_bn.py"):
        print("\nStopping because dependent_bn.py failed.")
        return

    # -------------------------------------------------------
    # Finished
    # -------------------------------------------------------

    print_header("ALL EXPERIMENTS COMPLETED")

    print("1. Bayesian Network inference     : COMPLETED")
    print("2. Naive Bayes classification     : COMPLETED")
    print("3. Dependent BN classification    : COMPLETED")

    print("\n")


if __name__ == "__main__":
    main()