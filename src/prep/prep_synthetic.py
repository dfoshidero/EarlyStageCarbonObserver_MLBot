import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""
1. Import synthetic datasets.
"""

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/raw')
export_dir = os.path.join(current_dir, '../../data/processed')

os.makedirs(export_dir, exist_ok=True)

fcbsData_30000A_path = os.path.join(data_dir, 'model/synthetic/220712FCBS_30000.csv')

fcbsData_30000A = pd.read_csv(fcbsData_30000A_path)

"""
2. Combine synthetic datasets.
"""

syntheticData = pd.concat([fcbsData_30000A])
print(syntheticData)
print("")
print("")

"""
2. Clean datasets and re-name variables to.
"""

# Replace nan values with "Other"
syntheticData.replace(np.nan, "Other", inplace=True)