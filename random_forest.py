from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import data_prep

# define the model (max_depth and n_estimators are set)
random_forest = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# train the model
random_forest.fit(data_prep.X_train, data_prep.y_train)

# prediction
y_pred_train = random_forest.predict(data_prep.X_train)
y_pred_test = random_forest.predict(data_prep.X_test)

# performance metrics
train_mse = mean_squared_error(data_prep.y_train, y_pred_train)
test_mse = mean_squared_error(data_prep.y_test, y_pred_test)

train_mae = mean_absolute_error(data_prep.y_train, y_pred_train)
test_mae = mean_absolute_error(data_prep.y_test, y_pred_test)

train_r2 = r2_score(data_prep.y_train, y_pred_train)
test_r2 = r2_score(data_prep.y_test, y_pred_test)

print(f"Random Forest (Adjusted) - Train MSE: {train_mse:.2f}")
print(f"Random Forest (Adjusted) - Test MSE: {test_mse:.2f}")

print(f"Random Forest (Adjusted) - Train MAE: {train_mae:.2f}")
print(f"Random Forest (Adjusted) - Test MAE: {test_mae:.2f}")

print(f"Random Forest (Adjusted) - Train R²: {train_r2:.2f}")
print(f"Random Forest (Adjusted) - Test R²: {test_r2:.2f}")

# graph
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data_prep.y_test, y_pred_test, alpha=0.6, color='dodgerblue', edgecolor='k', label='Tahminler')
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