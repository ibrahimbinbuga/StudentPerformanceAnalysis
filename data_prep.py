import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# dataset
data = pd.read_csv("StudentPerformanceFactors.csv")

# separating categorical and numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# missing value analysis
missing_summary = data.isnull().sum()
print("Missing values:")
print(missing_summary)

# filling in missing categorical data with mode
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# scaler
scaler = MinMaxScaler()
numeric_data_scaled = scaler.fit_transform(data[numeric_columns])

# filling in missing data with KNN
knn_imputer = KNNImputer(n_neighbors=5)
numeric_data_imputed = knn_imputer.fit_transform(numeric_data_scaled)

# returning data to original scale
numeric_data_restored = scaler.inverse_transform(numeric_data_imputed)
numeric_data_restored_df = pd.DataFrame(numeric_data_restored, columns=numeric_columns)

# updating numeric columns
data[numeric_columns] = numeric_data_restored_df

# final missing data check
missing_after = data.isnull().sum().sum()
print(f"Final missing values: {missing_after}")

data.to_csv("StudentPerformanceFactorsFilled.csv", index=False)


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

# features and target variable
X = data.drop(columns=[target_variable])  # features
y = data[target_variable]  # target variable

# creating interaction terms
X['Distance_Sleep_Interaction'] = X['Distance_from_Home'] * X['Sleep_Hours']
X['Previous_Scores_Motivation_Interaction'] = X['Previous_Scores'] * X['Motivation_Level']
X['Parental_Involvement_Motivation_Interaction'] = X['Parental_Involvement'] * X['Motivation_Level']
X['Peer_Influence_Motivation_Interaction'] = X['Peer_Influence'] * X['Motivation_Level']
X['Parental_Income_Access_To_Resources_Interaction'] = X['Family_Income'] * X['Access_to_Resources']
X['Internet_Access_Access_To_Resources_Interaction'] = X['Internet_Access'] * X['Access_to_Resources']
X['Physical_Activity_Motivation_Interaction'] = X['Physical_Activity'] * X['Motivation_Level']
X['Family_Income_Tutoring_Sessions_Interaction'] = X['Family_Income'] * X['Tutoring_Sessions']
X['Family_Income_School_Type_Interaction'] = X['Family_Income'] * X['School_Type']
X['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = X['Peer_Influence'] * X['Parental_Involvement'] * X['Motivation_Level']

from sklearn.model_selection import train_test_split

# split the data test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)