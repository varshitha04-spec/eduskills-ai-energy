# eduskills-ai-energy
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

