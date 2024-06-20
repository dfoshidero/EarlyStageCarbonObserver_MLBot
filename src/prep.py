"""
1. Import Libraries and Load Data   
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/raw')
export_dir = os.path.join(current_dir, '../data/processed')

os.makedirs(export_dir, exist_ok=True)

CARB_EN_MATS_PATH = os.path.join(data_dir, 'model/CarbEnMats_dataset.xlsx')

df = pd.read_excel(CARB_EN_MATS_PATH)


