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
    # Приведення тексту до нижнього регістру
    text = text.lower()
    # Видалення HTML тегів
    text = re.sub(r'<.*?>', '', text)
    # Видалення непотрібних символів
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

# Логістична регресія з масштабованими даними та розв'язувачем 'liblinear'
log_reg = LogisticRegression(max_iter=5000, solver='liblinear')
log_reg.fit(X_train_bow_scaled, y_train)
y_pred_bow_scaled = log_reg.predict(X_test_bow_scaled)
print("Logistic Regression (BOW, scaled) Accuracy: ", accuracy_score(y_test, y_pred_bow_scaled))
print(classification_report(y_test, y_pred_bow_scaled))

log_reg.fit(X_train_tfidf_scaled, y_train)
y_pred_tfidf_scaled = log_reg.predict(X_test_tfidf_scaled)
print("Logistic Regression (TF-IDF, scaled) Accuracy: ", accuracy_score(y_test, y_pred_tfidf_scaled))
print(classification_report(y_test, y_pred_tfidf_scaled))

# Логістична регресія з альтернативним розв'язувачем 'saga'
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
