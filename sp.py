import pandas as pd
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

target_variable = 'Exam_Score'
# Özellikler (features) ve hedef değişkeni (target) ayırma
X = data.drop(columns=[target_variable])  # Özellikler
y = data[target_variable]  # Hedef değişken

from sklearn.model_selection import train_test_split

# Veriyi eğitim (%80) ve test (%20) olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Sayısal özellikleri standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)