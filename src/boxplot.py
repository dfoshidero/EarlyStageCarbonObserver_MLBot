import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
carbenmats_df = pd.read_csv('path_to_cleaned_carbenmats.csv')  # Update the path as necessary

# List of relevant fields to plot against Total_Carbon
relevant_fields = ['Building_Use_Type', 'Continent', 'Country', 'Structure_Type']

# Set the size of the plots
plt.figure(figsize=(15, 10))

# Generate box plots for each relevant field
for i, field in enumerate(relevant_fields, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=field, y='Total_Carbon', data=carbenmats_df)
    plt.title(f'Box Plot of Total Carbon by {field}')
    plt.xticks(rotation=90)  # Rotate x labels for better visibility

plt.tight_layout()
plt.show()
