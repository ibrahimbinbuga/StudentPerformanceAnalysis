import shap
import pandas as pd
import data_prep
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# 1. XGBoost modelini oluştur
xgb_model = xgb.XGBRegressor(random_state=42)

# 2. Modeli eğit (Varsayılan parametrelerle)
xgb_model.fit(data_prep.X_train, data_prep.y_train)

# 3. Tahmin yap ve sonuçları değerlendir
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

# 4. Hiperparametre optimizasyonu (GridSearchCV ile)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(data_prep.X_train, data_prep.y_train)

# En iyi parametreleri bul
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 5. En iyi parametrelerle model eğitimi
best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_model.fit(data_prep.X_train, data_prep.y_train)

# 6. Tahmin yap ve sonuçları değerlendir
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

# SHAP için bir açıklayıcı oluştur
explainer = shap.Explainer(best_xgb_model, data_prep.X_train)
shap_values = explainer(data_prep.X_test)

# 1.1 Global Önem Grafiği
shap.summary_plot(shap_values, data_prep.X_test, plot_type="bar")

# 1.2 Detaylı Global Özet
shap.summary_plot(shap_values, data_prep.X_test)

# 1.3 Bireysel Tahmin Açıklaması
# Örneğin, X_test'in ilk satırı için
shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values[0].base_values,
                                     data=data_prep.X_test.iloc[0]))

# Grafik
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

