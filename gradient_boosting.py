from sklearn.metrics import mean_squared_error, r2_score

import data_prep
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    scoring='r2',
    cv=5
)
grid_search.fit(data_prep.X_train, data_prep.y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# define model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# train the model
gbr.fit(data_prep.X_train, data_prep.y_train)

# predictions
y_pred_train_gbr = gbr.predict(data_prep.X_train)
y_pred_test_gbr = gbr.predict(data_prep.X_test)

# performance metrics
train_mse_gbr = mean_squared_error(data_prep.y_train, y_pred_train_gbr)
test_mse_gbr = mean_squared_error(data_prep.y_test, y_pred_test_gbr)

train_mae_gbr = mean_absolute_error(data_prep.y_train, y_pred_train_gbr)
test_mae_gbr = mean_absolute_error(data_prep.y_test, y_pred_test_gbr)

train_r2_gbr = r2_score(data_prep.y_train, y_pred_train_gbr)
test_r2_gbr = r2_score(data_prep.y_test, y_pred_test_gbr)

print(f"Gradient Boosting - Train MSE: {train_mse_gbr:.2f}")
print(f"Gradient Boosting - Test MSE: {test_mse_gbr:.2f}")

print(f"Gradient Boosting - Train MAE: {train_mae_gbr:.2f}")
print(f"Gradient Boosting - Test MAE: {test_mae_gbr:.2f}")

print(f"Gradient Boosting - Train R²: {train_r2_gbr:.2f}")
print(f"Gradient Boosting - Test R²: {test_r2_gbr:.2f}")

import matplotlib.pyplot as plt

feature_importances = gbr.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(data_prep.X.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Özellik Önemi")
plt.title("Gradient Boosting Özellik Önemleri")
plt.show()

import shap

# SHAP
explainer = shap.Explainer(gbr, data_prep.X_train)
shap_values = explainer(data_prep.X_test)

# importance graph
shap.summary_plot(shap_values, data_prep.X_test, plot_type="bar")

shap.summary_plot(shap_values, data_prep.X_test)

shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values[0].base_values,
                                     data=data_prep.X_test.iloc[0]))

                                     # Grafik
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data_prep.y_test, y_pred_test_gbr, alpha=0.6, color='dodgerblue', edgecolor='k', label='Tahminler')
plt.plot([data_prep.y_test.min(), data_prep.y_test.max()], 
         [data_prep.y_test.min(), data_prep.y_test.max()], 
         color='red', linewidth=2, linestyle='--', label='Doğru Çizgi (y=x)')
plt.xlabel("Actual values", fontsize=12)
plt.ylabel("Prediction values", fontsize=12)
plt.title("Actual values vs Prediction values", fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.show()