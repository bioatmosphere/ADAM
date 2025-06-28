# A Belowground Productivity (BP) random forest model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression # For generating regression data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Generate a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=0.5, random_state=42)
# n_informative: number of features that are actually used to build the target

# Convert to a Pandas DataFrame for easier inspection (optional)
feature_names_reg = [f'feature_{i+1}' for i in range(X.shape[1])]
df_reg = pd.DataFrame(X, columns=feature_names_reg)
df_reg['target'] = y
print("Sample of the Regression dataset:")
print(df_reg.head())
print("\n")


# 2. Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: X_train={X_train_reg.shape}, y_train={y_train_reg.shape}")
print(f"Testing set shape: X_test={X_test_reg.shape}, y_test={y_test_reg.shape}")
print("\n")

# 3. Initialize the Random Forest Regressor
# Key hyperparameters are similar to the classifier
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)

# 4. Train the model
print("Training the Random Forest Regressor...")
rf_regressor.fit(X_train_reg, y_train_reg)
print("Training complete.\n")

# 5. Make predictions on the test set
y_pred_reg = rf_regressor.predict(X_test_reg)

# 6. Evaluate the model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse) # Root Mean Squared Error
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# Feature Importances for Regressor
importances_reg = rf_regressor.feature_importances_
feature_importance_df_reg = pd.DataFrame({'feature': feature_names_reg, 'importance': importances_reg})
feature_importance_df_reg = feature_importance_df_reg.sort_values(by='importance', ascending=False)

print("\nFeature Importances (Regressor):")
print(feature_importance_df_reg)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df_reg.head(10)) # Display top 10 features
plt.title('Feature Importances in Regression Task')
plt.show()

# Optional: Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], '--r', linewidth=2) # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Random Forest Regressor)')
plt.grid(True)
plt.show()


    
