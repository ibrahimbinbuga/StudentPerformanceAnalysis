# Student Performance Dataset

## Proje Tanımı
Bu proje, öğrenci performansını etkileyen çeşitli faktörleri analiz etmeyi amaçlayan bir makine öğrenimi projesidir. Proje, öğrencilerin derslerdeki başarılarını etkileyen faktörleri inceleyerek, bu faktörlere dayalı olarak başarı tahminleri yapmayı hedefler. Veri seti, çalışma alışkanlıkları, devamsızlık, ebeveyn katılımı gibi birçok faktörü içermektedir.

## İçindekiler
*   [Proje Özeti](#proje-ozeti)
*   [Veri Seti](#veri-seti)
*   [Model Eğitimi](#model-egitimi)
*   [Kullanım](#kullanım)
*   [Test ve Değerlendirme](#test-ve-degerlendirme)
*   [Katkı](#katki)


## Proje Özeti
Bu proje, öğrenci başarılarını etkileyen faktörleri incelemekte ve bu faktörlere dayanarak öğrenci performansını tahmin etmektedir. Öğrencilerin akademik başarılarını etkileyen faktörler, çalışma alışkanlıkları, ebeveyn katılımı, ders dışı etkinliklere katılım, uyku düzeni, önceki sınav notları, öğretmen kalitesi gibi bir dizi önemli faktörü içermektedir.

Projede kullanılan makine öğrenimi modelleri, bu faktörleri analiz ederek öğrenci başarıları üzerinde tahminler yapmaktadır.


## Veri Seti
Bu veri seti, öğrencilerin sınav başarılarını etkileyen çeşitli faktörleri kapsamlı bir şekilde incelemektedir. Veri seti, çalışma alışkanlıkları, devamsızlık, ebeveyn katılımı ve diğer akademik başarıyı etkileyen etmenleri içermektedir.



Veri seti, öğrencilerin çeşitli özelliklerini ve sınav sonuçlarını içerir. Bu özellikler, öğrencinin ders çalışma süresi, okula devam oranı, ebeveyn katılım düzeyi, kaynak erişimi, uyku saati gibi faktörleri kapsar. Bu faktörlerin her biri, öğrencinin sınav başarısını etkileyebilecek önemli unsurlardır.

| **Sütun Adı**              | **Açıklama**                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Hours_Studied**           | Haftada geçirilen çalışma saati sayısı.                                      |
| **Attendance**              | Katılımcı olunan derslerin yüzdesi.                                          |
| **Parental_Involvement**    | Ebeveynlerin eğitimdeki katılım düzeyi (Low, Medium, High).                  |
| **Access_to_Resources**     | Eğitim kaynaklarının erişilebilirliği (Low, Medium, High).                   |
| **Extracurricular_Activities** | Ekstra müfredat etkinliklerine katılım (Yes, No).                           |
| **Sleep_Hours**             | Ortalama uyku saati (günlük).                                                |
| **Previous_Scores**         | Önceki sınav sonuçları.                                                      |
| **Motivation_Level**        | Öğrencinin motivasyon düzeyi (Low, Medium, High).                            |
| **Internet_Access**         | İnternet erişimi (Yes, No).                                                  |
| **Tutoring_Sessions**       | Aylık katılınan özel ders sayısı.                                            |
| **Family_Income**           | Aile geliri düzeyi (Low, Medium, High).                                      |
| **Teacher_Quality**         | Öğretmenlerin kalitesi (Low, Medium, High).                                  |
| **School_Type**             | Okul türü (Public, Private).                                                 |
| **Peer_Influence**          | Arkadaş çevresinin akademik başarıya etkisi (Positive, Neutral, Negative).   |
| **Physical_Activity**       | Haftada yapılan ortalama fiziksel aktivite saati.                            |
| **Learning_Disabilities**   | Öğrenme güçlüklerinin varlığı (Yes, No).                                     |
| **Parental_Education_Level**| Ebeveynlerin eğitim düzeyi (High School, College, Postgraduate).             |
| **Distance_from_Home**      | Okula mesafe (Near, Moderate, Far).                                          |
| **Gender**                  | Öğrencinin cinsiyeti (Male, Female).                                         |
| **Exam_Score**              | Final sınav puanı.                                                           |


## Model Eğitimi
4 adet model eğitilmiştir.

****************************************************************************************************

### Linear Model
##### Why we tried it:
- We used it as the first step because it is simple and explainable. 
- We wanted to see the linear relationships between the target variable and the features.
##### Why we didn't choose it:
- Because we thought there are non-linear relationships between the features in our dataset.

##### Performans Değerlendirmesi:
- Mean Squared Error (MSE): 3.13
- Mean Absolute Error (MAE): 0.46
- R²: 0.77

****************************************************************************************************


### Random Forest Regressor
##### Why we tried it:
- It performs well on non-linear data and datasets with many features.
- It calculates feature importance.
- It helps prevent overfitting.
##### Why we didn't choose it:
- We thought there were more complex relationships between the features
- It lagged behind XGBoost in terms of performance metrics and speed

##### Performans Değerlendirmesi:
- Mean Squared Error (MSE): 4.61
- Mean Absolute Error (MAE): 1.13
- R²: 0.66
****************************************************************************************************
  
### Gradient Boosting Regressor
##### Why we tried it:
- Because of its capacity to learn more complex relationships.
- It works with the concept of incremental learning.
##### Why we didn't choose it:
- While its performance metrics were close to XGBoost, it lagged behind in flexibility and speed.

##### Performans Değerlendirmesi:
- Mean Squared Error (MSE): 3.96
- Mean Absolute Error (MAE): 0.83
- R²: 0.71

****************************************************************************************************
  
### XGBoost
##### Why we tried it:
- Because it provides explainability of the features' impact on the target variable.
- It also has fast computation capabilities.

#### Feature Intereaction Engineering
- Interactions between seemingly insignificant features can influence the features that have a greater impact on exam scores.
- Therefore, we established logical relationships between some features in the dataset. Example:

- Distance_from_Home -> Hours_Studied -> Exam_Score
-Family_Income -> School_Type -> Exam_Score
- Internet_Access -> Access_to_Resources -> Exam_Score
##### Why we chose it:
- It provided the best performance metrics.
- We observed an improvement in the model after feature interaction engineering.
- The model was able to learn both linear and non-linear relationships.
- The model was further improved after hyperparameter optimizaiton.

##### Performans Değerlendirmesi:
- Mean Squared Error (MSE): 3.46
- Mean Absolute Error (MAE): 0.68
- R²: 0.75

****************************************************************************************************
