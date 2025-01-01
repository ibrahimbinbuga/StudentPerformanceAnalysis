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
