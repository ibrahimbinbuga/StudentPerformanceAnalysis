import pandas as pd
import xgboost_model

# Öğrencinin verisini bir sözlük olarak tanımlayın
student_data = {
    'Hours_Studied': 30,                    # Haftalık çalışma saati
    'Attendance': 100,                       # Katılım oranı (%)
    'Parental_Involvement': 2,              # Aile katılımı (3: High)
    'Access_to_Resources': 2,               # Kaynaklara erişim (2: Medium)
    'Extracurricular_Activities': 1,        # Ekstra aktivitelerde katılım (1: Yes)
    'Sleep_Hours': 7,                       # Uyku saati
    'Previous_Scores': 100,                  # Önceki sınav puanı
    'Motivation_Level': 2,                  # Motivasyon seviyesi (2: Medium)
    'Internet_Access': 1,                   # İnternet erişimi (1: Yes)
    'Tutoring_Sessions': 10,                # Katılınan özel ders sayısı
    'Family_Income': 2,                     # Aile geliri (3: High)
    'Teacher_Quality': 2,                   # Öğretmen kalitesi (2: Medium)
    'School_Type': 1,                       # Okul türü (1: Private)
    'Peer_Influence': 1,                    # Akran etkisi (1: Positive)
    'Physical_Activity': 5,                 # Haftalık fiziksel aktivite saati
    'Learning_Disabilities': 0,             # Öğrenme güçlüğü (0: No)
    'Parental_Education_Level': 3,          # Ebeveynin eğitim durumu (3: Postgraduate)
    'Distance_from_Home': 0,                # Okul mesafesi (2: Moderate)
    'Gender': 0,       
}

# Etkileşim terimlerini hesapla
student_data['Distance_Sleep_Interaction'] = student_data['Distance_from_Home'] * student_data['Sleep_Hours']
student_data['Previous_Scores_Motivation_Interaction'] = student_data['Previous_Scores'] * student_data['Motivation_Level']
student_data['Parental_Involvement_Motivation_Interaction'] = student_data['Parental_Involvement'] * student_data['Motivation_Level']
student_data['Peer_Influence_Motivation_Interaction'] = student_data['Peer_Influence'] * student_data['Motivation_Level']
student_data['Parental_Income_Access_To_Resources_Interaction'] = student_data['Family_Income'] * student_data['Access_to_Resources']
student_data['Internet_Access_Access_To_Resources_Interaction'] = student_data['Internet_Access'] * student_data['Access_to_Resources']
student_data['Physical_Activity_Motivation_Interaction'] = student_data['Physical_Activity'] * student_data['Motivation_Level']
student_data['Family_Income_Tutoring_Sessions_Interaction'] = student_data['Family_Income'] * student_data['Tutoring_Sessions']
student_data['Family_Income_School_Type_Interaction'] = student_data['Family_Income'] * student_data['School_Type']
student_data['Peer_Influence_Parental_Involvement_Motivation_Interaction'] = student_data['Peer_Influence'] * student_data['Parental_Involvement'] * student_data['Motivation_Level']

# Bu veriyi pandas DataFrame formatına dönüştürün
student_df = pd.DataFrame([student_data])

# Etkileşimli özellikleri tahmin verisi üzerinde de ekleyin
student_df = xgboost_model.add_interactions(student_df)

# Modelin tahmin yapması için student_df'yi kullanın
student_prediction = xgboost_model.best_xgb_model.predict(student_df)


# Öğrenci verisi ile tahmin yapın
student_prediction = xgboost_model.best_xgb_model.predict(student_df)

# Tahmin edilen sınav puanını yazdır
print(f"Öğrencinin tahmin edilen sınav puanı: {student_prediction[0]:.2f}")
