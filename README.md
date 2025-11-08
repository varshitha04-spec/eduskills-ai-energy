# eduskills-ai-energy
# this contains the whole code
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
# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# 2. Data Quality Solutions - Imputation
# Impute numerical missing values with KNN
num_imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute categorical with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Handle boolean columns
boolean_cols = ['Perovskite_deposition_quenching_induced_crystallisation', 'Encapsulation', 'JV_measured']
for col in boolean_cols:
    df[col] = df[col].astype(bool).astype(int)  # Convert to 0/1

# Drop rows with missing PCE if still any (after imputation)
df = df.dropna(subset=['JV_default_PCE'])
df.head(5)
# 3. Feature Engineering
# Parse Perovskite_composition_short_form (e.g., "CsFAMAPbBrI")
# Define known ions
a_ions = ['Cs', 'FA', 'MA', 'Rb', 'K']  # Example list, extend as needed
b_ions = ['Pb', 'Sn']
c_ions = ['I', 'Br', 'Cl']

def parse_short_form(formula):
    features = {}
    for ion in a_ions + b_ions + c_ions:
        features[f'has_{ion}'] = 1 if ion in formula else 0
    return features

parsed_df = df['Perovskite_composition_short_form'].apply(parse_short_form).apply(pd.Series)
print(parsed_df)
df = pd.concat([df, parsed_df], axis=1)

# One-hot encode categorical features (select top ones to avoid high dimensionality)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=10)
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())
df = pd.concat([df, encoded_df], axis=1).drop(categorical_cols, axis=1)

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols[:-4]] = scaler.fit_transform(df[numerical_cols[:-4]])  # Exclude JV params

# Interaction features (example)
# df['thickness_annealing_temp'] = df['Perovskite_thickness'] * df['Perovskite_deposition_thermal_annealing_temperature']

df.head(5)
y = df['JV_default_PCE']
X = df.drop(['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE'], axis=1)

# 4. Performance Prediction (PCE)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nRandom Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("Random Forest R2:", r2_score(y_test, rf_pred))
# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
print("RF CV R2 Scores:", cv_scores.mean())
!pip install xgboost --upgrade
import xgboost as xgb

# Convert data to DMatrix format (required for xgb.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.5,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'seed': 42
}

# Train model with early stopping
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1500,
    evals=[(dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False
)
# Neural Network with Torch
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Prepare data
train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleNN(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    nn_pred = model(torch.tensor(X_test.values, dtype=torch.float32)).numpy().flatten()
print("\nNeural Network RMSE:", np.sqrt(mean_squared_error(y_test, nn_pred)))
print("Neural Network R2:", r2_score(y_test, nn_pred))

# Feature Importance with SHAP (using XGBoost)

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Feature Importance')
plt.show()
# 5. Anomaly Detection
# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(X)
df['anomaly_iso'] = anomalies
print("\nNumber of Anomalies (Isolation Forest):", (anomalies == -1).sum())

# One-Class SVM
oc_svm = OneClassSVM(nu=0.05)
anomalies_svm = oc_svm.fit_predict(X)
df['anomaly_svm'] = anomalies_svm

# Autoencoder with Torch for anomaly
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_size))

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = Autoencoder(X.shape[1])
optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train AE
ae_dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32), torch.tensor(X.values, dtype=torch.float32))
ae_loader = DataLoader(ae_dataset, batch_size=32, shuffle=True)

for epoch in range(50):
    ae_model.train()
    for batch_x, _ in ae_loader:
        optimizer.zero_grad()
        output = ae_model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()

# Reconstruction error
ae_model.eval()
with torch.no_grad():
    recon = ae_model(torch.tensor(X.values, dtype=torch.float32)).numpy()
re_errors = np.mean((X.values - recon)**2, axis=1)
threshold = np.percentile(re_errors, 95)
df['anomaly_ae'] = (re_errors > threshold).astype(int)
print("Number of Anomalies (Autoencoder):", df['anomaly_ae'].sum())
# 6. Unsupervised Learning - Clustering
# KMeans on features
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Correlate clusters with PCE
print("\nPCE by Cluster:")
print(df.groupby('cluster')['JV_default_PCE'].mean())

# Visualization with t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_emb = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_emb[:,0], y=tsne_emb[:,1], hue=df['JV_default_PCE'], palette='viridis')
plt.title('t-SNE Visualization Colored by PCE')
plt.show()
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Assuming df, X, and targets are defined
targets = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
y_multi = df[targets]

# Split data for multi-output model
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=0.2, random_state=42)

# Multi-output XGBoost
multi_xgb = MultiOutputRegressor(
    xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        random_state=42
    )
)
multi_xgb.fit(X_train_m, y_train_m)
multi_pred = multi_xgb.predict(X_test_m)

# Print RMSE for each parameter
print("\nJoint Prediction RMSE (per parameter):", np.sqrt(mean_squared_error(y_test_m, multi_pred, multioutput='raw_values')))

# Train a single-output model for PCE only
y_pce = df['JV_default_PCE']  # Single target for PCE
X_train_pce, X_test_pce, y_train_pce, y_test_pce = train_test_split(X, y_pce, test_size=0.2, random_state=42)

single_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42
)
single_xgb.fit(X_train_pce, y_train_pce)
single_pred = single_xgb.predict(X_test_pce)

# Compare R² scores for PCE
multi_pce_r2 = r2_score(y_test_m['JV_default_PCE'], multi_pred[:, 3])  # PCE is the 4th column (index 3)
single_pce_r2 = r2_score(y_test_pce, single_pred)

print("Multi-output PCE R²:", multi_pce_r2)
print("Single-output PCE R²:", single_pce_r2)
print("Improvement in PCE R²:", multi_pce_r2 - single_pce_r2)



# # Investigate anomalies (example: high PCE anomalies)
# # high_pce_anomalies = df[(df['anomaly_iso'] == -1) & (df['JV_default_PCE'] > df['JV_default_PCE'].quantile(0.95))]
# # print("\nHigh PCE Anomalies:")
# # print(high_pce_anomalies[['Perovskite_composition_short_form', 'JV_default_PCE']])



# # 8. Uncertainty (example with bootstrap for RF)
# def bootstrap_ci(model, X, n_boot=100):
#     preds = np.array([model.predict(X) for _ in range(n_boot)]).T  # Note: this assumes model has randomness or retrain
#     ci_lower = np.percentile(preds, 2.5, axis=1)
#     ci_upper = np.percentile(preds, 97.5, axis=1)
#     return ci_lower, ci_upper

# # For simplicity, use RF with bootstrap=True
# rf_boot = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=42)
# rf_boot.fit(X_train, y_train)
# ci_lower, ci_upper = bootstrap_ci(rf_boot, X_test[:10])  # Example on first 10
# print("\nExample Confidence Intervals for PCE Predictions:")
# for i in range(10):
#     print(f"Pred: {rf_pred[i]:.2f}, CI: [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]")

# # 9. Error Analysis
# errors = np.abs(y_test - xgb_pred)
# # high_error_idx = errors.nlargest(10).index
# # print("\nHigh Error Samples:")
# # print(df.loc[high_error_idx, ['Perovskite_composition_short_form', 'Cell_architecture', 'JV_default_PCE']])

# # Filter high PCE anomalies
# high_pce_anomalies = df[(df['anomaly_iso'] == -1) & (df['JV_default_PCE'] > df['JV_default_PCE'].quantile(0.95))]



# # End of Script

# print("\nAnalysis Complete.")
