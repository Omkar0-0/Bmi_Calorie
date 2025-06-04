# train_bmi_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import pickle

# Load dataset
df = pd.read_csv("bmi_dataset_simplified_classification.csv")

# Encode gender and BMI category
gender_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])  # Female=0, Male=1
df['BMI_Category'] = bmi_encoder.fit_transform(df['BMI_Category'])

# Features and targets
X = df[['Gender', 'Height_cm', 'Weight_kg']]
y_bmi = df['BMI_Category']
y_cal = df['Recommended_Calories']

# Train BMI Classifier
X_train_bmi, X_test_bmi, y_bmi_train, y_bmi_test = train_test_split(X, y_bmi, test_size=0.2, random_state=42)
bmi_clf = RandomForestClassifier(n_estimators=100, random_state=42)
bmi_clf.fit(X_train_bmi, y_bmi_train)
bmi_accuracy = accuracy_score(y_bmi_test, bmi_clf.predict(X_test_bmi))
print(f"BMI Classification Accuracy: {bmi_accuracy:.2f}")

# Train Calorie Regressor
X_train_cal, X_test_cal, y_cal_train, y_cal_test = train_test_split(X, y_cal, test_size=0.2, random_state=42)
cal_reg = RandomForestRegressor(n_estimators=100, random_state=42)
cal_reg.fit(X_train_cal, y_cal_train)
cal_r2 = r2_score(y_cal_test, cal_reg.predict(X_test_cal))
print(f"Calorie Prediction R² Score: {cal_r2:.2f}")

# Save models with metrics
with open("bmi_category_model.pkl", "wb") as f1:
    pickle.dump((bmi_clf, bmi_accuracy), f1)

with open("calorie_predictor_model.pkl", "wb") as f2:
    pickle.dump((cal_reg, cal_r2), f2)

with open("bmi_label_encoder.pkl", "wb") as f3:
    pickle.dump(bmi_encoder, f3)

print("✅ Models saved with accuracy and R² score")
