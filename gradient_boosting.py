from sklearn.metrics import mean_squared_error, r2_score

import data_prep
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

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

# Modeli tanımlama
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Modeli eğitme
gbr.fit(data_prep.X_train, data_prep.y_train)

# Tahminler
y_pred_train_gbr = gbr.predict(data_prep.X_train)
y_pred_test_gbr = gbr.predict(data_prep.X_test)

# Performans metrikleri
train_mse_gbr = mean_squared_error(data_prep.y_train, y_pred_train_gbr)
test_mse_gbr = mean_squared_error(data_prep.y_test, y_pred_test_gbr)

train_r2_gbr = r2_score(data_prep.y_train, y_pred_train_gbr)
test_r2_gbr = r2_score(data_prep.y_test, y_pred_test_gbr)

print(f"Gradient Boosting - Train MSE: {train_mse_gbr:.2f}")
print(f"Gradient Boosting - Test MSE: {test_mse_gbr:.2f}")
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

# SHAP için bir açıklayıcı oluştur
explainer = shap.Explainer(gbr, data_prep.X_train)
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