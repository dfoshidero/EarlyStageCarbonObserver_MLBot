import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../../data/raw")
export_dir = os.path.join(current_dir, "../../data/processed")

os.makedirs(export_dir, exist_ok=True)

CLF_EMBODIED_CARBON_PATH = os.path.join(
    data_dir, "model/CLF Embodied Carbon_Cleaned.csv"
)

clf_df = pd.read_csv(CLF_EMBODIED_CARBON_PATH)

"""
2. Clean the Datasets
"""
#### CLF
# Convert relevant columns to numeric, forcing errors to NaN
cols_to_convert = [
    "Minimum Building Area in Square Meters",
    "Maximum Building Area in Square Meters",
    "Minimum Building Storeys",
    "Maximum Building Storeys",
]

for col in cols_to_convert:
    clf_df[col] = pd.to_numeric(clf_df[col], errors="coerce")

# Replace np.inf and NaN with the maximum observed value in the respective columns
for col in cols_to_convert:
    clf_df[col].replace(np.inf, np.nan, inplace=True)
    max_value = clf_df[col].max(skipna=True)
    clf_df[col].fillna(max_value, inplace=True)

# Calculate average area and storeys
clf_df["Average Building Area in Square Meters"] = (
    clf_df["Minimum Building Area in Square Meters"]
    + clf_df["Maximum Building Area in Square Meters"]
) / 2
clf_df["Average Building Storeys"] = (
    clf_df["Minimum Building Storeys"] + clf_df["Maximum Building Storeys"]
) / 2

# Drop the minimum and maximum columns if they are no longer needed
clf_df = clf_df.drop(
    columns=[
        "Minimum Building Area in Square Meters",
        "Maximum Building Area in Square Meters",
        "Minimum Building Storeys",
        "Maximum Building Storeys",
    ]
)

# Select relevant columns
clf_df = clf_df[
    [
        "Building Type",
        "Building Use",
        "Building Location Region",
        "Building New or Renovation",
        "Average Building Storeys",
        "Embodied Carbon Life Cycle Assessment Area Per Square Meter",
    ]
]

clf_df = clf_df.dropna()

clf_df = clf_df.rename(
    columns={
        "Building Type": "Building Use Type",
        "Building Use": "Building Use Subtype",
        "Building Location Region": "Continent",
        "Building New or Renovation": "Building Project Type",
        "Average Building Storeys": "Floors Above Ground",
        "Embodied Carbon Life Cycle Assessment Area Per Square Meter": "Total Embodied Carbon PER m2",
    }
)

# Replace all values in "Building_Use_Type" that are not equal to "Residential" with "Non-residential"
clf_df["Building Use Type"] = clf_df["Building Use Type"].apply(
    lambda x: "Residential" if x == "Residential" else "Non-residential"
)

"""
3. Remove Outliers
"""
# Calculate Q1, Q3, and IQR for Total_Embodied_Carbon
Q1 = clf_df["Total Embodied Carbon PER m2"].quantile(0.25)
Q3 = clf_df["Total Embodied Carbon PER m2"].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
clf_df = clf_df[
    (clf_df["Total Embodied Carbon PER m2"] >= lower_bound)
    & (clf_df["Total Embodied Carbon PER m2"] <= upper_bound)
]

"""
4. Save dataframe to CSV for inspection.
"""
clf_df_PATH = os.path.join(export_dir, "inspect/cleaned_clf.csv")
clf_df.to_csv(clf_df_PATH, index=False)
clf_df.info()

"""
5. Label encode categorical data for ML use.
"""
# Define categorical columns
categorical_cols = [
    "Building Use Type",
    "Building Use Subtype",
    "Continent",
    "Building Project Type",
]

# Label encode categorical columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    clf_df[col] = label_encoders[col].fit_transform(clf_df[col])

"""
6. Save dataframe to CSV for modeling.
"""
clf_df_PATH = os.path.join(export_dir, "encoded/encoded_clf.csv")
clf_df.to_csv(clf_df_PATH, index=False)
clf_df.info()
