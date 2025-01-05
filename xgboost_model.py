import shap
import pandas as pd
import data_prep
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# a function to add interactive features
def add_interactions(X):
    X['Previous_Scores_Motivation_Interaction'] = X['Previous_Scores'] * X['Motivation_Level']
    X['Family_Income_Access_To_Resources_Interaction'] = X['Family_Income'] * X['Access_to_Resources']
    X['Internet_Access_Access_To_Resources_Interaction'] = X['Internet_Access'] * X['Access_to_Resources']
    X['Physical_Activity_Motivation_Interaction'] = X['Physical_Activity'] * X['Motivation_Level']
    X['Family_Income_Tutoring_Sessions_Interaction'] = X['Family_Income'] * X['Tutoring_Sessions']
    X['Family_Income_School_Type_Interaction'] = X['Family_Income'] * X['School_Type']
    X['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = X['Peer_Influence'] * X['Parental_Involvement'] * X['Motivation_Level']
    X['Distance_Sleep_Interaction'] = X['Distance_from_Home'] * X['Sleep_Hours']
    X['Parental_Involvement_Motivation_Interaction'] = X['Parental_Involvement'] * X['Motivation_Level']
    X['Peer_Influence_Motivation_Interaction'] = X['Peer_Influence'] * X['Motivation_Level']
    return X

data_prep.X_train = add_interactions(data_prep.X_train)
data_prep.X_test = add_interactions(data_prep.X_test)

# xgboost model
xgb_model = xgb.XGBRegressor(random_state=42)

# train the model
xgb_model.fit(data_prep.X_train, data_prep.y_train)

# prediction
y_train_pred = xgb_model.predict(data_prep.X_train)
y_test_pred = xgb_model.predict(data_prep.X_test)

train_mse = mean_squared_error(data_prep.y_train, y_train_pred)
test_mse = mean_squared_error(data_prep.y_test, y_test_pred)
train_r2 = r2_score(data_prep.y_train, y_train_pred)
test_r2 = r2_score(data_prep.y_test, y_test_pred)
train_mae = mean_absolute_error(data_prep.y_train, y_train_pred)
test_mae = mean_absolute_error(data_prep.y_test, y_test_pred)

print(f"XGBoost - Train MSE: {train_mse:.2f}")
print(f"XGBoost - Test MSE: {test_mse:.2f}")

print(f"XGBoost - Train MAE: {train_mae:.2f}")
print(f"XGBoost - Test MAE: {test_mae:.2f}")

print(f"XGBoost - Train R²: {train_r2:.2f}")
print(f"XGBoost - Test R²: {test_r2:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data_prep.y_test, y_test_pred, alpha=0.6, color='dodgerblue', edgecolor='k', label='Tahminler')
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

# hyperparameter optimization (with GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(data_prep.X_train, data_prep.y_train)

# best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# model training with best parameters
best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_model.fit(data_prep.X_train, data_prep.y_train)

# prediction
y_train_pred_best = best_xgb_model.predict(data_prep.X_train)
y_test_pred_best = best_xgb_model.predict(data_prep.X_test)

train_mse_best = mean_squared_error(data_prep.y_train, y_train_pred_best)
test_mse_best = mean_squared_error(data_prep.y_test, y_test_pred_best)
train_r2_best = r2_score(data_prep.y_train, y_train_pred_best)
test_r2_best = r2_score(data_prep.y_test, y_test_pred_best)
train_mae_best = mean_absolute_error(data_prep.y_train, y_train_pred_best)
test_mae_best = mean_absolute_error(data_prep.y_test, y_test_pred_best)

print(f"XGBoost (Tuned) - Train MSE: {train_mse_best:.2f}")
print(f"XGBoost (Tuned) - Test MSE: {test_mse_best:.2f}")

print(f"XGBoost (Tuned) - Train MAE: {train_mae_best:.2f}")
print(f"XGBoost (Tuned) - Test MAE: {test_mae_best:.2f}")

print(f"XGBoost (Tuned) - Train R²: {train_r2_best:.2f}")
print(f"XGBoost (Tuned) - Test R²: {test_r2_best:.2f}")

# SHAP
explainer = shap.Explainer(best_xgb_model, data_prep.X_train)
shap_values = explainer(data_prep.X_test)

shap.summary_plot(shap_values, data_prep.X_test, plot_type="bar")

shap.summary_plot(shap_values, data_prep.X_test)

shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values[0].base_values,
                                     data=data_prep.X_test.iloc[0]))

# graph
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data_prep.y_test, y_test_pred_best, alpha=0.6, color='dodgerblue', edgecolor='k', label='Tahminler')
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

