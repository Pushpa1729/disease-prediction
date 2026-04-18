import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

os.makedirs('models', exist_ok=True)

# ========== 1. DIABETES ==========
print("Training Diabetes Model...")
df = pd.read_csv('dataset/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Diabetes Accuracy: {model.score(X_test, y_test)*100:.2f}%")
pickle.dump(model, open('models/diabetes_model.pkl', 'wb'))
pickle.dump(scaler, open('models/diabetes_scaler.pkl', 'wb'))

# ========== 2. HEART DISEASE ==========
print("Training Heart Disease Model...")
df = pd.read_csv('dataset/heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Heart Disease Accuracy: {model.score(X_test, y_test)*100:.2f}%")
pickle.dump(model, open('models/heart_model.pkl', 'wb'))
pickle.dump(scaler, open('models/heart_scaler.pkl', 'wb'))

# ========== 3. PARKINSONS ==========
print("Training Parkinsons Model...")
df = pd.read_csv('dataset/parkinsons.csv')
df = df.drop('name', axis=1)
X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Parkinsons Accuracy: {model.score(X_test, y_test)*100:.2f}%")
pickle.dump(model, open('models/parkinsons_model.pkl', 'wb'))
pickle.dump(scaler, open('models/parkinsons_scaler.pkl', 'wb'))

# ========== 4. LIVER DISEASE ==========
print("Training Liver Disease Model...")
df = pd.read_csv('dataset/indian_liver_patient.csv')
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = df.dropna()
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
X = df.drop('Dataset', axis=1)
y = df['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Liver Disease Accuracy: {model.score(X_test, y_test)*100:.2f}%")
pickle.dump(model, open('models/liver_model.pkl', 'wb'))
pickle.dump(scaler, open('models/liver_scaler.pkl', 'wb'))

# ========== 5. KIDNEY DISEASE ==========
print("Training Kidney Disease Model...")
df = pd.read_csv('dataset/kidney_disease.csv')
df = df.drop('id', axis=1)
df = df.replace({'yes': 1, 'no': 0, 'ckd': 1, 'notckd': 0,
                 'normal': 1, 'abnormal': 0, 'present': 1,
                 'notpresent': 0, 'good': 1, 'poor': 0})
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.median(numeric_only=True))
X = df.drop('classification', axis=1)
y = df['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Kidney Disease Accuracy: {model.score(X_test, y_test)*100:.2f}%")
pickle.dump(model, open('models/kidney_model.pkl', 'wb'))
pickle.dump(scaler, open('models/kidney_scaler.pkl', 'wb'))

print("\n✅ All 5 Models Trained & Saved Successfully!")