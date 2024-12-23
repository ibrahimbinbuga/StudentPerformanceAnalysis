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