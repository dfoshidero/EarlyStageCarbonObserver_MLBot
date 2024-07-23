import pandas as pd


# Function to get unique values and column names from an Excel file as a dictionary, ignoring specified columns
def get_unique_values_and_columns_dict(file_path, ignore_columns=[]):
    # Read the Excel file
    df = pd.read_csv(file_path)

    # Create a dictionary to store unique values for each column
    unique_values_dict = {}

    # Iterate over each column in the DataFrame
    for col in df.columns:
        if col not in ignore_columns:
            unique_values_dict[col] = df[col].unique().tolist()

    return unique_values_dict


# Example usage
file_path = (
    "data/processed/inspect/cleaned_synthetic.csv"  # Replace with your actual file path
)
ignore_columns = [
    "Gross Internal Area (m2)",
    "Building Perimeter (m)",
    "Building Footprint (m2)",
    "Building Width (m)",
    "Floor-to-Floor Height (m)",
    "Storeys Above Ground",
    "Storeys Below Ground",
    "Glazing Ratio (%)",
    "Embodied Carbon (kgCO2e/m2)",
]  # Replace with the columns you want to ignore
unique_vals_dict = get_unique_values_and_columns_dict(file_path, ignore_columns)

# Print the dictionary
print("Unique Values Dictionary (excluding ignored columns):")
for col, values in unique_vals_dict.items():
    print(f"{col}: {values}")
