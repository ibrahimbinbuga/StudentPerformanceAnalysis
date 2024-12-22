import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Veriyi yükleme
data = pd.read_csv("StudentPerformanceFactors.csv")

# 2. Kategorik ve sayısal sütunları ayırma
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# 3. Eksik veri analizi
missing_summary = data.isnull().sum()
print("Eksik veri sayısı:")
print(missing_summary)

# 4. Eksik kategorik verileri mod ile doldurma
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# 5. Eksik sayısal verileri KNN ile doldurmak için hazırlık
scaler = MinMaxScaler()
numeric_data_scaled = scaler.fit_transform(data[numeric_columns])

# 6. KNN ile eksik verileri doldurma
knn_imputer = KNNImputer(n_neighbors=5)
numeric_data_imputed = knn_imputer.fit_transform(numeric_data_scaled)

# 7. Veriyi orijinal ölçeğe geri döndürme
numeric_data_restored = scaler.inverse_transform(numeric_data_imputed)
numeric_data_restored_df = pd.DataFrame(numeric_data_restored, columns=numeric_columns)

# 8. Sayısal sütunları güncelleme
data[numeric_columns] = numeric_data_restored_df

# 9. Son eksik veri kontrolü
missing_after = data.isnull().sum().sum()
print(f"Son eksik veri sayısı: {missing_after}")

data.to_csv("StudentPerformanceFactorsFilled.csv", index=False)

import seaborn as sns

# Sayısal verilerin dağılımı (Histogramlar)
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f'{col} Dağılımı')
    plt.show()

# Boxplot ile aykırı değerlerin kontrolü
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[col])
    plt.title(f'{col} Boxplot')
    plt.show()

# Kategorik verilerin dağılımı (Bar grafik)
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data[col])
    plt.title(f'{col} Dağılımı')
    plt.show()

# Korelasyon ısı haritası
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Sayısal Sütunlar Arasındaki Korelasyon')
plt.show()

# Pairplot ile ilişkilerin görselleştirilmesi
sns.pairplot(data[numeric_columns])
plt.show()

# Son eksik veri kontrolü
missing_after = data.isnull().sum().sum()
print(f"Son eksik veri sayısı: {missing_after}")



priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
distance_mapping = {'Near':0, 'Moderate':1, 'Far':2}
parental_education_mapping = {'High School':0, 'College':1, 'Postgraduate':2}
yes_or_no_mapping = {'Yes': 1, 'No': 0}
peer_influence_mapping = {'Negative':0, 'Neutral':1,'Positive':2}
gender_mapping = {'Female':0, 'Male':1}
school_type_mapping = {'Public': 0, 'Private': 1}

data['Parental_Involvement'] = data['Parental_Involvement'].map(priority_mapping)
data['Access_to_Resources'] = data['Access_to_Resources'].map(priority_mapping)
data['Motivation_Level'] = data['Motivation_Level'].map(priority_mapping)
data['Family_Income'] = data['Family_Income'].map(priority_mapping)
data['Teacher_Quality'] = data['Teacher_Quality'].map(priority_mapping)

data['Extracurricular_Activities'] = data['Extracurricular_Activities'].map(yes_or_no_mapping)
data['Internet_Access'] = data['Internet_Access'].map(yes_or_no_mapping)
data['Learning_Disabilities'] = data['Learning_Disabilities'].map(yes_or_no_mapping)

data['Distance_from_Home'] = data['Distance_from_Home'].map(distance_mapping)

data['Parental_Education_Level'] = data['Parental_Education_Level'].map(parental_education_mapping)

data['Peer_Influence'] = data['Peer_Influence'].map(peer_influence_mapping)

data['Gender'] = data['Gender'].map(gender_mapping)
data['School_Type'] = data['School_Type'].map(school_type_mapping)

data.to_csv("StudentPerformanceMapped.csv", index=False)

categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                       'Parental_Education_Level', 'Distance_from_Home', 'Gender']
plt.figure(figsize=(16, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

target_variable = 'Exam_Score'
# Özellikler (features) ve hedef değişkeni (target) ayırma
X = data.drop(columns=[target_variable])  # Özellikler
y = data[target_variable]  # Hedef değişken

from sklearn.model_selection import train_test_split

# Veriyi eğitim (%70) ve test (%30) olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Modeli tanımlama
linear_model = LinearRegression()

# Lineer regresyon modelini eğitme
linear_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = linear_model.predict(X_test)

# Performans metriği (MSE ve R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Lineer Regresyon - Mean Squared Error (MSE): {mse:.2f}")
print(f"Lineer Regresyon - R²: {r2:.2f}")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})
print(coefficients.sort_values(by="Coefficient", ascending=False))

# Modeli tanımlama (max_depth ve n_estimators ayarlandı)
random_forest = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Modeli eğitme
random_forest.fit(X_train, y_train)

# Tahminler
y_pred_train = random_forest.predict(X_train)
y_pred_test = random_forest.predict(X_test)

# Performans metrikleri
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Random Forest (Adjusted) - Train MSE: {train_mse:.2f}")
print(f"Random Forest (Adjusted) - Test MSE: {test_mse:.2f}")
print(f"Random Forest (Adjusted) - Train R²: {train_r2:.2f}")
print(f"Random Forest (Adjusted) - Test R²: {test_r2:.2f}")

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
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Modeli tanımlama
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Modeli eğitme
gbr.fit(X_train, y_train)

# Tahminler
y_pred_train_gbr = gbr.predict(X_train)
y_pred_test_gbr = gbr.predict(X_test)

# Performans metrikleri
train_mse_gbr = mean_squared_error(y_train, y_pred_train_gbr)
test_mse_gbr = mean_squared_error(y_test, y_pred_test_gbr)

train_r2_gbr = r2_score(y_train, y_pred_train_gbr)
test_r2_gbr = r2_score(y_test, y_pred_test_gbr)

print(f"Gradient Boosting - Train MSE: {train_mse_gbr:.2f}")
print(f"Gradient Boosting - Test MSE: {test_mse_gbr:.2f}")
print(f"Gradient Boosting - Train R²: {train_r2_gbr:.2f}")
print(f"Gradient Boosting - Test R²: {test_r2_gbr:.2f}")

import matplotlib.pyplot as plt

feature_importances = gbr.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(X.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Özellik Önemi")
plt.title("Gradient Boosting Özellik Önemleri")
plt.show()

import shap

# SHAP için bir açıklayıcı oluştur
explainer = shap.Explainer(gbr, X_train)
shap_values = explainer(X_test)

# 1.1 Global Önem Grafiği
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 1.2 Detaylı Global Özet
shap.summary_plot(shap_values, X_test)

# 1.3 Bireysel Tahmin Açıklaması
# Örneğin, X_test'in ilk satırı için
shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values[0].base_values,
                                     data=X_test.iloc[0]))


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# 1. XGBoost modelini oluştur
xgb_model = xgb.XGBRegressor(random_state=42)

# 2. Modeli eğit (Varsayılan parametrelerle)
xgb_model.fit(X_train, y_train)

# 3. Tahmin yap ve sonuçları değerlendir
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"XGBoost - Train MSE: {train_mse:.2f}")
print(f"XGBoost - Test MSE: {test_mse:.2f}")
print(f"XGBoost - Train R²: {train_r2:.2f}")
print(f"XGBoost - Test R²: {test_r2:.2f}")

# 4. Hiperparametre optimizasyonu (GridSearchCV ile)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi parametreleri bul
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 5. En iyi parametrelerle model eğitimi
best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_model.fit(X_train, y_train)

# 6. Tahmin yap ve sonuçları değerlendir
y_train_pred_best = best_xgb_model.predict(X_train)
y_test_pred_best = best_xgb_model.predict(X_test)

train_mse_best = mean_squared_error(y_train, y_train_pred_best)
test_mse_best = mean_squared_error(y_test, y_test_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

print(f"XGBoost (Tuned) - Train MSE: {train_mse_best:.2f}")
print(f"XGBoost (Tuned) - Test MSE: {test_mse_best:.2f}")
print(f"XGBoost (Tuned) - Train R²: {train_r2_best:.2f}")
print(f"XGBoost (Tuned) - Test R²: {test_r2_best:.2f}")

# SHAP için bir açıklayıcı oluştur
explainer = shap.Explainer(gbr, X_train)
shap_values = explainer(X_test)

# 1.1 Global Önem Grafiği
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 1.2 Detaylı Global Özet
shap.summary_plot(shap_values, X_test)

# 1.3 Bireysel Tahmin Açıklaması
# Örneğin, X_test'in ilk satırı için
shap.waterfall_plot(shap.Explanation(values=shap_values[0].values,
                                     base_values=shap_values[0].base_values,
                                     data=X_test.iloc[0]))

X['Distance_Sleep_Interaction'] = X['Distance_From_Home'] * X['Sleep_Time']
X['Previous_Score_Motivation_Interaction'] = X['Previous_Score'] * X['Motivation_Level']
X['Parental_Involvement_Motivation_Interaction'] = X['Parental_Involvement'] * X['Motivation_Level']
X['Peer_Influence_Motivation_Interaction'] = X['Peer_Influence'] * X['Motivation_Level']
X['Parental_Income_Access_To_Resources_Interaction'] = X['Parental_Income'] * X['Access_to_Resources']
X['Internet_Access_Access_To_Resources_Interaction'] = X['Internet_Access'] * X['Access_to_Resources']
X['Physical_Activity_Motivation_Interaction'] = X['Physical_Activity'] * X['Motivation_Level']
X['Family_Income_Tutoring_Sessions_Interaction'] = X['Family_Income'] * X['Tutoring_Sessions']
X['Family_Income_School_Type_Interaction'] = X['Family_Income'] * X['School_Type']
X['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = X['Peer_Influence'] * X['Parental_Involvement'] * X['Motivation_Level']