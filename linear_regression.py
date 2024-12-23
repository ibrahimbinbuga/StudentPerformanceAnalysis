import data_prep
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Modeli tanımlama
linear_model = LinearRegression()

# Lineer regresyon modelini eğitme
linear_model.fit(data_prep.X_train, data_prep.y_train)

# Test verisi üzerinde tahmin yapma
y_pred = linear_model.predict(data_prep.X_test)

# Performans metriği (MSE ve R²)
mse = mean_squared_error(data_prep.y_test, y_pred)
r2 = r2_score(data_prep.y_test, y_pred)
mae = mean_absolute_error(data_prep.y_test, y_pred)

print(f"Lineer Regresyon - Mean Squared Error (MSE): {mse:.2f}")
print(f"Lineer Regresyon - Mean Absolute Error (MAE): {mae:.2f}")
print(f"Lineer Regresyon - R²: {r2:.2f}")

coefficients = pd.DataFrame({
    "Feature": data_prep.X.columns,
    "Coefficient": linear_model.coef_
})
print(coefficients.sort_values(by="Coefficient", ascending=False))

# Grafik
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data_prep.y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k', label='Tahminler')
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