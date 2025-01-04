# Student Performance Factor

## Project Description
This project is a machine learning project aimed at analyzing various factors affecting student performance. The project seeks to examine factors influencing students' academic success and make predictions based on these factors. The dataset includes many variables such as study habits, attendance, and parental involvement.
****************************************************************************************************
## İçindekiler
*   [Project Summary](#project-summary)
*   [Dataset](#dataset)
*   [Data Preparation](#data-preparation)
*   [Model Training](#model-traning)
*   [Usage](#usage)
*   [Test ve Değerlendirme](#test-ve-degerlendirme)
*   [Contribution](#contribution)
****************************************************************************************************

## Project Summary
This project analyzes factors affecting student success and predicts student performance based on these factors. The significant factors influencing students' academic success include study habits, parental involvement, participation in extracurricular activities, sleep patterns, previous exam scores, and teacher quality.

The machine learning models used in the project analyze these factors to predict academic performance.

****************************************************************************************************
## Dataset
This dataset comprehensively examines various factors affecting students' exam success. It includes variables such as study habits, attendance, parental involvement, and other factors influencing academic performance.

The dataset contains various student attributes and exam results. These attributes include study time, school attendance rate, parental involvement level, resource accessibility, sleep hours, and more. Each of these factors plays an essential role in impacting exam success.

| **Sütun Adı**              | **Açıklama**                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Hours_Studied**           | Number of study hours per week.                                      |
| **Attendance**              | 	Percentage of classes attended.                                          |
| **Parental_Involvement**    | Level of parental involvement in education (Low, Medium, High).                  |
| **Access_to_Resources**     | Accessibility of educational resources (Low, Medium, High).                |
| **Extracurricular_Activities** | 	Participation in extracurricular activities (Yes, No).                          |
| **Sleep_Hours**             | 	Average sleep hours per day.                                             |
| **Previous_Scores**         | Previous exam scores.                                             |
| **Motivation_Level**        | Student's motivation level (Low, Medium, High).                         |
| **Internet_Access**         |	Internet access (Yes, No).                                           |
| **Tutoring_Sessions**       | Number of private tutoring sessions per month.                                     |
| **Family_Income**           | 	Family income level (Low, Medium, High).                              |
| **Teacher_Quality**         | Teacher quality (Low, Medium, High).                                |
| **School_Type**             | 	Type of school (Public, Private).                                                 |
| **Peer_Influence**          |Influence of peers on academic success (Positive, Neutral, Negative).   |
| **Physical_Activity**       | Average hours of physical activity per week.                           |
| **Learning_Disabilities**   | Presence of learning disabilities (Yes, No).                                   |
| **Parental_Education_Level**| Parental education level (High School, College, Postgraduate).           |
| **Distance_from_Home**      | Distance from home to school (Near, Moderate, Far).                           |
| **Gender**                  |Gender of the student (Male, Female).                                       |
| **Exam_Score**              |Final exam score.                                                          |
****************************************************************************************************
## Data Preparation
The data preparation process was carried out in the following steps:

#### Loading the Dataset:

- The dataset StudentPerformanceFactors.csv was loaded into a Pandas DataFrame for further analysis.
#### Separation of Columns:

- Numerical and categorical columns were separated based on their data types.
#### Handling Missing Data:

- For categorical columns, missing values were replaced with the mode of each column.
- For numerical columns, the missing values were imputed using K-Nearest Neighbors (KNN) after scaling the data to normalize the values.
#### Outlier Analysis:

- Boxplots were created for each numerical column to visualize potential outliers.
- No outlier removal was performed at this stage, but it was noted for further optimization.
#### Data Visualization:

- Distribution plots and histograms were generated for numerical columns to understand their spread and distribution.
- Bar plots were created for categorical columns to visualize category frequencies.
#### Correlation Analysis:

- A heatmap of the correlation matrix was plotted to identify significant relationships between numerical features.
#### Mapping Categorical Values:

- Categorical columns were encoded into numerical values using custom mappings for better compatibility with machine learning algorithms.
#### Feature Engineering:

- Interaction terms were created between key variables.
#### Splitting the Data:

- The dataset was split into training (70%) and testing (30%) subsets to prepare for model training and evaluation.
#### Final Preprocessed Data:

- The preprocessed data was saved into StudentPerformanceMapped.csv for reproducibility and reference.
****************************************************************************************************
## Model Training
Four models were trained.

****************************************************************************************************

#### Linear Model
##### Reasons for Selection:
- Initially utilized due to its simplicity and explainability.
- It provided insights into the linear relationships between the target variable and the features.
##### Reason for Exclusion:
- Non-linear relationships between the features in the dataset were identified.
##### Performance Evaluation:
- Mean Squared Error (MSE): 3.13
- Mean Absolute Error (MAE): 0.46
- R²: 0.77

****************************************************************************************************


#### Random Forest Regressor
##### Reasons for Selection:
- Suitable for non-linear data and datasets with multiple features.
- Calculates feature importance.
- Helps mitigate overfitting.
##### Reason for Exclusion:
- Despite its robust performance metrics, it was outperformed by XGBoost in terms of accuracy and speed.

##### Performance Evaluation:
- Mean Squared Error (MSE): 4.61
- Mean Absolute Error (MAE): 1.13
- R²: 0.66
****************************************************************************************************
  
#### Gradient Boosting Regressor
##### Reasons for Selection:
- Chosen for its capacity to model complex relationships.
- Operates on the principle of incremental learning.
##### Reason for Exclusion:
- Although its performance metrics were close to XGBoost, it fell short in terms of flexibility and speed.
##### Performance Evaluation:
- Mean Squared Error (MSE): 3.96
- Mean Absolute Error (MAE): 0.83
- R²: 0.71

****************************************************************************************************
  
#### XGBoost
##### Reasons for Selection
- Selected due to its capability to explain the impact of features on the target variable.
- Offers efficient computation.
#### Feature Intereaction Engineering
- Interactions between seemingly insignificant features can influence the features that have a greater impact on exam scores.
- Logical relationships between key variables in the dataset were established. Example:
- Distance_from_Home -> Hours_Studied -> Exam_Score
- Family_Income -> School_Type -> Exam_Score
- Internet_Access -> Access_to_Resources -> Exam_Score
##### Reasons for Final Selection:
- Demonstrated superior performance metrics.
- Significant improvement was observed after implementing feature interaction engineering.
- Effectively learned both linear and non-linear relationships.
- Further enhancement was achieved through hyperparameter optimization.
##### Performance Evaluation:
- Mean Squared Error (MSE): 3.46
- Mean Absolute Error (MAE): 0.68
- R²: 0.75

****************************************************************************************************


## Installation and Usage
#### Prerequisites:
To run this project, you need the following Python libraries:
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```
You can install the required dependencies using pip:


```pip install -r requirements.txt```
### Running the Project:
- Clone the repository:

```
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction 
```
- Download the dataset:
  
Download the "Student Performance Factors" dataset from Kaggle and place it in the data/ directory.

- Train the Model:
  
Run the following script to train the model and evaluate its performance:

```
python xgboost_model.py
```
- Make Predictions:

Once the model is trained, use it to make predictions on new data:

```
python predict.py
```
- Visualize Results:
  
Use the following script to visualize the results:
```
python graphs.py
```
