# app_bmi.py

import streamlit as st
import numpy as np
import pickle

# Streamlit config
st.set_page_config(page_title="BMI & Calorie Predictor", page_icon="🍏", layout="centered")

# Load models with metrics
bmi_model, bmi_accuracy = pickle.load(open("bmi_category_model.pkl", "rb"))
calorie_model, cal_r2 = pickle.load(open("calorie_predictor_model.pkl", "rb"))

# Title and metrics
st.title("🌿 BMI & Daily Calorie Predictor")
st.markdown(f"🧠 **BMI Classification Accuracy**: {bmi_accuracy * 100:.2f}%")
st.markdown(f"🔥 **Calorie Prediction Accuracy (R²):** {cal_r2 * 100:.2f}%")
st.markdown("Enter your **gender**, **height**, and **weight** to get your BMI and daily calorie recommendation.")

# BMI Categories Info
st.markdown("""
**BMI Categories:**
- Underweight: BMI < 18.5
- Normal: BMI 18.5 - 24.9
- Overweight: BMI 25 - 29.9
- Obesity: BMI ≥ 30
""")

# Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, format="%.1f")
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, format="%.1f")

# Predict
if st.button("🔍 Predict"):
    bmi = weight / ((height / 100) ** 2)
    gender_encoded = 1 if gender == "Male" else 0
    input_data = np.array([[gender_encoded, height, weight]])

    cal_pred = calorie_model.predict(input_data)[0]

    st.success(f"**Calculated BMI:** {bmi:.2f}")
    st.warning(f"**Recommended Daily Calories:** {int(cal_pred)} kcal")
