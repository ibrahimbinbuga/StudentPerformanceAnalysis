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