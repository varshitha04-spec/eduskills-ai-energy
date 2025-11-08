# eduskills-ai-energy
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

