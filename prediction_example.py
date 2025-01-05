import pandas as pd
import xgboost_model

# ********************** First Student **********************
# define the data for the first student as a dictionary
first_student = {
    'Hours_Studied': 30,                    # Weekly study hours
    'Attendance': 100,                      # Attendance rate (%)
    'Parental_Involvement': 2,              # Parental involvement (2: High)
    'Access_to_Resources': 2,               # Access to resources (2: High)
    'Extracurricular_Activities': 1,        # Participation in extracurricular activities (1: Yes)
    'Sleep_Hours': 7,                       # Sleep hours
    'Previous_Scores': 100,                 # Previous exam score
    'Motivation_Level': 2,                  # Motivation level (2: High)
    'Internet_Access': 1,                   # Internet access (1: Yes)
    'Tutoring_Sessions': 10,                # Number of tutoring sessions
    'Family_Income': 2,                     # Family income (2: High)
    'Teacher_Quality': 2,                   # Teacher quality (2: High)
    'School_Type': 1,                       # School type (1: Private)
    'Peer_Influence': 1,                    # Peer influence (1: Positive)
    'Physical_Activity': 5,                 # Weekly physical activity hours
    'Learning_Disabilities': 0,             # Learning disabilities (0: No)
    'Parental_Education_Level': 2,          # Parent's education level (2: Postgraduate)
    'Distance_from_Home': 0,                # Distance from home to school (0: Near)
    'Gender': 0,                            # Gender (0: Male)
}

# calculate interaction terms
first_student['Distance_Sleep_Interaction'] = first_student['Distance_from_Home'] * first_student['Sleep_Hours']
first_student['Previous_Scores_Motivation_Interaction'] = first_student['Previous_Scores'] * first_student['Motivation_Level']
first_student['Parental_Involvement_Motivation_Interaction'] = first_student['Parental_Involvement'] * first_student['Motivation_Level']
first_student['Peer_Influence_Motivation_Interaction'] = first_student['Peer_Influence'] * first_student['Motivation_Level']
first_student['Parental_Income_Access_To_Resources_Interaction'] = first_student['Family_Income'] * first_student['Access_to_Resources']
first_student['Internet_Access_Access_To_Resources_Interaction'] = first_student['Internet_Access'] * first_student['Access_to_Resources']
first_student['Physical_Activity_Motivation_Interaction'] = first_student['Physical_Activity'] * first_student['Motivation_Level']
first_student['Family_Income_Tutoring_Sessions_Interaction'] = first_student['Family_Income'] * first_student['Tutoring_Sessions']
first_student['Family_Income_School_Type_Interaction'] = first_student['Family_Income'] * first_student['School_Type']
first_student['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = first_student['Peer_Influence'] * first_student['Parental_Involvement'] * first_student['Motivation_Level']


first_student_df = pd.DataFrame([first_student])

first_student_df = xgboost_model.add_interactions(first_student_df)

first_student_prediction = xgboost_model.best_xgb_model.predict(first_student_df)

print(f"The predicted exam score for the first student is: {first_student_prediction[0]:.2f}")

# ********************** Second Student **********************
# define the data for the second student as a dictionary
second_student = {
    'Hours_Studied': 5,                     # Weekly study hours
    'Attendance': 10,                       # Attendance rate (%)
    'Parental_Involvement': 0,              # Parental involvement (0: Low)
    'Access_to_Resources': 0,               # Access to resources (0: Low)
    'Extracurricular_Activities': 0,        # Participation in extracurricular activities (1: Yes)
    'Sleep_Hours': 12,                      # Sleep hours
    'Previous_Scores': 30,                  # Previous exam score
    'Motivation_Level': 0,                  # Motivation level (0: Low)
    'Internet_Access': 0,                   # Internet access (0: No)
    'Tutoring_Sessions': 1,                 # Number of tutoring sessions
    'Family_Income': 0,                     # Family income (0: Low)
    'Teacher_Quality': 0,                   # Teacher quality (0: Low)
    'School_Type': 0,                       # School type (0: Public)
    'Peer_Influence': 0,                    # Peer influence (0: Negative)
    'Physical_Activity': 1,                 # Weekly physical activity hours
    'Learning_Disabilities': 1,             # Learning disabilities (1: Yes)
    'Parental_Education_Level': 0,          # Parent's education level (0: High School)
    'Distance_from_Home': 2,                # Distance from home to school (2: Far)
    'Gender': 0,                            # Gender (0: Male)
}

# calculate interaction terms
second_student['Distance_Sleep_Interaction'] = second_student['Distance_from_Home'] * second_student['Sleep_Hours']
second_student['Previous_Scores_Motivation_Interaction'] = second_student['Previous_Scores'] * second_student['Motivation_Level']
second_student['Parental_Involvement_Motivation_Interaction'] = second_student['Parental_Involvement'] * second_student['Motivation_Level']
second_student['Peer_Influence_Motivation_Interaction'] = second_student['Peer_Influence'] * second_student['Motivation_Level']
second_student['Parental_Income_Access_To_Resources_Interaction'] = second_student['Family_Income'] * second_student['Access_to_Resources']
second_student['Internet_Access_Access_To_Resources_Interaction'] = second_student['Internet_Access'] * second_student['Access_to_Resources']
second_student['Physical_Activity_Motivation_Interaction'] = second_student['Physical_Activity'] * second_student['Motivation_Level']
second_student['Family_Income_Tutoring_Sessions_Interaction'] = second_student['Family_Income'] * second_student['Tutoring_Sessions']
second_student['Family_Income_School_Type_Interaction'] = second_student['Family_Income'] * second_student['School_Type']
second_student['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = second_student['Peer_Influence'] * second_student['Parental_Involvement'] * second_student['Motivation_Level']


second_student_df = pd.DataFrame([second_student])

second_student_df = xgboost_model.add_interactions(second_student_df)

second_student_prediction = xgboost_model.best_xgb_model.predict(second_student_df)

print(f"The predicted exam score for the second student is: {second_student_prediction[0]:.2f}")