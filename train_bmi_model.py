# train_bmi_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("bmi_dataset_simplified_classification.csv")

# Encode gender and BMI category
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Female=0, Male=1
df['BMI_Category'] = LabelEncoder().fit_transform(df['BMI_Category'])  # Encode BMI categories

# Features and targets
X = df[['Gender', 'Height_cm', 'Weight_kg']]
y_bmi = df['BMI_Category']
y_cal = df['Recommended_Calories']

# Split and train classifier for BMI category
X_train, X_test, y_bmi_train, y_bmi_test = train_test_split(X, y_bmi, test_size=0.2, random_state=42)
bmi_clf = RandomForestClassifier(n_estimators=100, random_state=42)
bmi_clf.fit(X_train, y_bmi_train)

# Split and train regressor for calorie prediction
X_train_cal, X_test_cal, y_cal_train, y_cal_test = train_test_split(X, y_cal, test_size=0.2, random_state=42)
cal_reg = RandomForestRegressor(n_estimators=100, random_state=42)
cal_reg.fit(X_train_cal, y_cal_train)

# Save models
with open("bmi_category_model.pkl", "wb") as f1:
    pickle.dump(bmi_clf, f1)

with open("calorie_predictor_model.pkl", "wb") as f2:
    pickle.dump(cal_reg, f2)

with open("bmi_label_encoder.pkl", "wb") as f3:
    pickle.dump(LabelEncoder().fit(df['BMI_Category']), f3)

print("Models saved: bmi_category_model.pkl and calorie_predictor_model.pkl")
