import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Завантаження даних
df = pd.read_csv('IMDB Dataset.csv')

# Препроцесинг даних
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['review'] = df['review'].apply(preprocess_text)

# Розбиття на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Перетворення тексту в числові ознаки (BOW)
vectorizer_bow = CountVectorizer(max_features=10000, min_df=5)
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# Перетворення тексту в числові ознаки (TF-IDF)
vectorizer_tfidf = TfidfVectorizer(max_features=10000, min_df=5)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Масштабування даних
scaler = StandardScaler(with_mean=False)  # with_mean=False оскільки дані розріджені
X_train_bow_scaled = scaler.fit_transform(X_train_bow)
X_test_bow_scaled = scaler.transform(X_test_bow)
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)

# Логістична регресія з ще більшою кількістю ітерацій
log_reg = LogisticRegression(max_iter=5000, solver='newton-cg')
log_reg.fit(X_train_bow, y_train)
y_pred_bow = log_reg.predict(X_test_bow)
print("Logistic Regression (BOW) Accuracy: ", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

log_reg.fit(X_train_tfidf, y_train)
y_pred_tfidf = log_reg.predict(X_test_tfidf)
print("Logistic Regression (TF-IDF) Accuracy: ", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

# Логістична регресія з масштабованими даними та солвером 'liblinear'
log_reg = LogisticRegression(max_iter=5000, solver='liblinear')
log_reg.fit(X_train_bow_scaled, y_train)
y_pred_bow_scaled = log_reg.predict(X_test_bow_scaled)
print("Logistic Regression (BOW, scaled) Accuracy: ", accuracy_score(y_test, y_pred_bow_scaled))
print(classification_report(y_test, y_pred_bow_scaled))

log_reg.fit(X_train_tfidf_scaled, y_train)
y_pred_tfidf_scaled = log_reg.predict(X_test_tfidf_scaled)
print("Logistic Regression (TF-IDF, scaled) Accuracy: ", accuracy_score(y_test, y_pred_tfidf_scaled))
print(classification_report(y_test, y_pred_tfidf_scaled))

# Логістична регресія з альтернативним солвером 'saga'
log_reg = LogisticRegression(solver='saga', max_iter=5000)
log_reg.fit(X_train_bow, y_train)
y_pred_bow_saga = log_reg.predict(X_test_bow)
print("Logistic Regression (BOW, solver='saga') Accuracy: ", accuracy_score(y_test, y_pred_bow_saga))
print(classification_report(y_test, y_pred_bow_saga))

log_reg.fit(X_train_tfidf, y_train)
y_pred_tfidf_saga = log_reg.predict(X_test_tfidf)
print("Logistic Regression (TF-IDF, solver='saga') Accuracy: ", accuracy_score(y_test, y_pred_tfidf_saga))
print(classification_report(y_test, y_pred_tfidf_saga))

# SGD Classifier
sgd = SGDClassifier(max_iter=5000)
sgd.fit(X_train_bow, y_train)
y_pred_bow_sgd = sgd.predict(X_test_bow)
print("SGD Classifier (BOW) Accuracy: ", accuracy_score(y_test, y_pred_bow_sgd))
print(classification_report(y_test, y_pred_bow_sgd))

sgd.fit(X_train_tfidf, y_train)
y_pred_tfidf_sgd = sgd.predict(X_test_tfidf)
print("SGD Classifier (TF-IDF) Accuracy: ", accuracy_score(y_test, y_pred_tfidf_sgd))
print(classification_report(y_test, y_pred_tfidf_sgd))

# Ridge Classifier
ridge = RidgeClassifier()
ridge.fit(X_train_bow, y_train)
y_pred_bow_ridge = ridge.predict(X_test_bow)
print("Ridge Classifier (BOW) Accuracy: ", accuracy_score(y_test, y_pred_bow_ridge))
print(classification_report(y_test, y_pred_bow_ridge))

ridge.fit(X_train_tfidf, y_train)
y_pred_tfidf_ridge = ridge.predict(X_test_tfidf)
print("Ridge Classifier (TF-IDF) Accuracy: ", accuracy_score(y_test, y_pred_tfidf_ridge))
print(classification_report(y_test, y_pred_tfidf_ridge))



""" My results: 
Logistic Regression (BOW) Accuracy:  0.8734
              precision    recall  f1-score   support

    negative       0.88      0.87      0.87      4961
    positive       0.87      0.88      0.87      5039

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000

Logistic Regression (TF-IDF) Accuracy:  0.8958
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.89      0.91      0.90      5039

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

Logistic Regression (BOW, scaled) Accuracy:  0.835
              precision    recall  f1-score   support

    negative       0.83      0.83      0.83      4961
    positive       0.84      0.84      0.84      5039

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.84      0.83      0.84     10000

Logistic Regression (TF-IDF, scaled) Accuracy:  0.8339
              precision    recall  f1-score   support

    negative       0.83      0.83      0.83      4961
    positive       0.84      0.83      0.83      5039

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.83      0.83      0.83     10000

Logistic Regression (BOW, solver='saga') Accuracy:  0.8905
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.89      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Logistic Regression (TF-IDF, solver='saga') Accuracy:  0.8952
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.89      0.91      0.90      5039

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

SGD Classifier (BOW) Accuracy:  0.875
              precision    recall  f1-score   support

    negative       0.90      0.84      0.87      4961
    positive       0.85      0.91      0.88      5039

    accuracy                           0.88     10000
   macro avg       0.88      0.87      0.87     10000
weighted avg       0.88      0.88      0.87     10000

SGD Classifier (TF-IDF) Accuracy:  0.8955
              precision    recall  f1-score   support

    negative       0.90      0.89      0.89      4961
    positive       0.89      0.90      0.90      5039

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000

Ridge Classifier (BOW) Accuracy:  0.8452
              precision    recall  f1-score   support

    negative       0.85      0.84      0.84      4961
    positive       0.84      0.85      0.85      5039

    accuracy                           0.85     10000
   macro avg       0.85      0.85      0.85     10000
weighted avg       0.85      0.85      0.85     10000

Ridge Classifier (TF-IDF) Accuracy:  0.8914
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000


Process finished with exit code 0


Використання TF-IDF як текстового представлення дало кращі результати у порівнянні з BOW.
Логістична регресія та SGD класифікатори показали схожі результати, причому обидва дали високу точність.
Ridge Classifier також показав хороші результати, але дещо нижчі, ніж інші моделі з TF-IDF.
Масштабування даних не покращило результати для логістичної регресії.
На основі цих результатів, для задачі класифікації відгуків найкращими варіантами є Logistic Regression або SGD Classifier з використанням TF-IDF.
"""
