import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
import_dir = os.path.join(current_dir, '../data/processed')

CLF_EMBODIED_CARBON_PATH = os.path.join(import_dir, 'cleaned_clf.csv')
CARB_EN_MATS_PATH = os.path.join(import_dir, 'cleaned_carbenmats.csv')
BECD_PATH = os.path.join(import_dir, 'cleaned_becd.csv')

# Load the datasets
clf_df = pd.read_csv(CLF_EMBODIED_CARBON_PATH)
carbenmats_df = pd.read_csv(CARB_EN_MATS_PATH)
becd_df = pd.read_csv(BECD_PATH)

# Plotting the boxplots for each dataset separately

# CLF dataset
plt.figure(figsize=(10, 6))
sns.boxplot(y=clf_df['Total_Embodied_Carbon_PER_m2'])
plt.title('CLF Embodied Carbon')
plt.ylabel('Total Embodied Carbon per m2')
plt.show()


# CarbEnMats dataset
plt.figure(figsize=(10, 6))
sns.boxplot(y=carbenmats_df['Total_Embodied_Carbon_PER_m2'])
plt.title('CarbEnMats Embodied Carbon')
plt.ylabel('Total Embodied Carbon per m2')
plt.show()

# BECD dataset
plt.figure(figsize=(10, 6))
sns.boxplot(y=becd_df['Total_Embodied_Carbon_PER_m2'])
plt.title('BECD Embodied Carbon')
plt.ylabel('Total Embodied Carbon per m2')
plt.show()
