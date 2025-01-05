import data_prep
import seaborn as sns
import matplotlib.pyplot as plt

# distribution of numerical data
for col in data_prep.numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data_prep.data[col], kde=True, bins=20)
    plt.title(f'{col} Dağılımı')
    plt.show()

# checking for outliers with boxplot
for col in data_prep.numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data_prep.data[col])
    plt.title(f'{col} Boxplot')
    plt.show()



# correlation heatmap
correlation_matrix = data_prep.data[data_prep.numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Sayısal Sütunlar Arasındaki Korelasyon')
plt.show()

# Pairplot
sns.pairplot(data_prep.data[data_prep.numeric_columns])
plt.show()