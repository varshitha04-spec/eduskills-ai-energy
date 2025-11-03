# eduskills-ai-energy
# categorical_cols = [
#     "Cell_stack_sequence", "Cell_architecture", "Substrate_stack_sequence",
#     "ETL_stack_sequence", "ETL_deposition_procedure", "Perovskite_composition_a_ions",
#     "Perovskite_composition_b_ions", "Perovskite_composition_c_ions",
#     "Perovskite_composition_short_form", "Perovskite_additives_compounds",
#     "Perovskite_deposition_procedure", "Perovskite_deposition_solvents",
#     "HTL_stack_sequence", "HTL_deposition_procedure", "Backcontact_stack_sequence",
#     "Backcontact_deposition_procedure", "Encapsulation_stack_sequence"
# ]
# numerical_cols = [
#     "Perovskite_thickness", "Perovskite_deposition_thermal_annealing_temperature",
#     "Perovskite_deposition_thermal_annealing_time", "Backcontact_thickness_list",
#     "Perovskite_deposition_number_of_deposition_steps"
# ]

# boolean_cols = [
#     "Perovskite_deposition_quenching_induced_crystallisation", "JV_measured", "Encapsulation"
# ]
# X_transformed = model.named_steps["preprocessor"].transform(X)

# X_transformed
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import warnings
# Load the data
try:
    df = pd.read_csv("Solar Cell data set.csv")
except FileNotFoundError:
    print("Error: 'Solar Cell data set.csv' not found. Please upload the file or provide the correct path.")
    # You might want to exit or handle this error differently depending on your needs
    # For now, we'll assume the file will be uploaded and the next cells can run
    pass


# Display basic info
if 'df' in locals(): # Check if df was successfully loaded
    print("Dataset Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
# 1. Exploratory Data Analysis (EDA)
# Numerical distributions
numerical_cols = ['Perovskite_thickness',"Perovskite_additives_concentrations", 'Perovskite_deposition_thermal_annealing_temperature',
                  'Perovskite_deposition_thermal_annealing_time', 'Backcontact_thickness_list',
                  'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']

def clean_numerical(value):
    if pd.isna(value) or value in ["nan", "None", "", "unknown"]:
        return np.nan
    if isinstance(value, str):
        # Normalize separators
        for sep in ['|', ',', ';']:
            if sep in value:
                values = [val.strip() for val in value.split(sep) if val.strip().lower() not in ['nan', 'none', '']]
                try:
                    values = [float(val) for val in values if val]
                    return np.mean(values) if values else np.nan
                except (ValueError, TypeError):
                    return np.nan
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    return float(value)


for col in numerical_cols:
    df[col] = df[col].apply(clean_numerical).astype(float)

df[numerical_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle('Numerical Feature Distributions')
plt.show()

# Categorical unique counts
categorical_cols = ['Cell_stack_sequence', 'Cell_architecture', 'Substrate_stack_sequence',
                    'ETL_stack_sequence', 'ETL_deposition_procedure', 'Perovskite_composition_a_ions',
                    'Perovskite_composition_b_ions', 'Perovskite_composition_c_ions',
                    'Perovskite_composition_short_form', 'Perovskite_additives_compounds',
                    'Perovskite_deposition_procedure', 'Perovskite_deposition_solvents',
                    'HTL_stack_sequence', 'HTL_deposition_procedure', 'Backcontact_stack_sequence',
                    'Backcontact_deposition_procedure', 'Encapsulation_stack_sequence']

for col in categorical_cols:
    print(f"\nUnique values in {col}: {df[col].nunique()}")


