import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
num_samples = 600

# Generate random skill ratings between 1 and 10
data = {
    "Python": np.random.randint(1, 11, size=num_samples),
    "Java": np.random.randint(1, 11, size=num_samples),
    "SQL": np.random.randint(1, 11, size=num_samples),
    "Machine_Learning": np.random.randint(1, 11, size=num_samples),
    "Data_Analysis": np.random.randint(1, 11, size=num_samples),
    "Communication": np.random.randint(1, 11, size=num_samples),
}

df = pd.DataFrame(data)


# Assign realistic target career roles based on skill dominance
def assign_career(row):
    if (
        row["Machine_Learning"] >= 7
        and row["Python"] >= 7
        and row["Data_Analysis"] >= 6
    ):
        return "Machine Learning Engineer"
    elif row["Data_Analysis"] >= 7 and row["SQL"] >= 7 and row["Python"] >= 5:
        return "Data Analyst"
    elif row["Java"] >= 7 and row["SQL"] >= 6 and row["Python"] >= 5:
        return "Software Engineer (Backend)"
    elif row["Communication"] >= 8 and row["Data_Analysis"] >= 6:
        return "Data Product Manager"
    else:
        return "Junior Data Associate"


df["Target_Career"] = df.apply(assign_career, axis=1)

# Save to dataset.csv
df.to_csv("dataset.csv", index=False)
print("Dataset successfully generated and saved to dataset.csv")