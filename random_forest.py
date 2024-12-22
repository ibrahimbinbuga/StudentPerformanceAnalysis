from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import data_prep

# Modeli tanımlama (max_depth ve n_estimators ayarlandı)
random_forest = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Modeli eğitme
random_forest.fit(data_prep.X_train, data_prep.y_train)

# Tahminler
y_pred_train = random_forest.predict(data_prep.X_train)
y_pred_test = random_forest.predict(data_prep.X_test)

# Performans metrikleri
train_mse = mean_squared_error(data_prep.y_train, y_pred_train)
test_mse = mean_squared_error(data_prep.y_test, y_pred_test)

train_r2 = r2_score(data_prep.y_train, y_pred_train)
test_r2 = r2_score(data_prep.y_test, y_pred_test)

print(f"Random Forest (Adjusted) - Train MSE: {train_mse:.2f}")
print(f"Random Forest (Adjusted) - Test MSE: {test_mse:.2f}")
print(f"Random Forest (Adjusted) - Train R²: {train_r2:.2f}")
print(f"Random Forest (Adjusted) - Test R²: {test_r2:.2f}")