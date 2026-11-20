from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

BASE_DIR = Path(__file__).resolve().parent
CLF_PATH = BASE_DIR / "models" / "classifier_model.joblib"
REG_PATH = BASE_DIR / "models" / "regressor_model.joblib"


@st.cache_resource
def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg


st.title("🩺 Diabetes Prediction")

if not CLF_PATH.exists() or not REG_PATH.exists():
    st.error("Run `python train.py` first to create the models.")
    st.stop()

clf_bundle, reg_bundle = load_models()

age = st.number_input("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Female", "Male"])
ethnicity = st.selectbox("Ethnicity", ["White", "Asian", "Black", "Hispanic"])
bmi = st.number_input("BMI", 10.0, 60.0, 29.4)
waist = st.number_input("Waist Circumference", 40.0, 180.0, 94.8)
fbg = st.number_input("Fasting Blood Glucose", 50.0, 350.0, 134.8)
bp_sys = st.number_input("Blood Pressure Systolic", 70, 240, 134)
bp_dia = st.number_input("Blood Pressure Diastolic", 40, 150, 90)
chol_tot = st.number_input("Cholesterol Total", 80.0, 450.0, 225.0)
chol_hdl = st.number_input("Cholesterol HDL", 10.0, 140.0, 55.0)
chol_ldl = st.number_input("Cholesterol LDL", 20.0, 300.0, 134.0)
ggt = st.number_input("GGT", 5.0, 300.0, 55.0)
urate = st.number_input("Serum Urate", 1.0, 20.0, 5.5)
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
calories = st.number_input("Dietary Intake Calories", 800, 7000, 2740)
alcohol = st.selectbox("Alcohol Consumption", ["Moderate", "Heavy", "None/Unknown"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
fam_hist = st.selectbox("Family History of Diabetes", [0, 1])
gest_diab = st.selectbox("Previous Gestational Diabetes", [0, 1])

if st.button("Predict"):
    row = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Ethnicity": ethnicity,
        "BMI": bmi,
        "Waist_Circumference": waist,
        "Fasting_Blood_Glucose": fbg,
        "Blood_Pressure_Systolic": bp_sys,
        "Blood_Pressure_Diastolic": bp_dia,
        "Cholesterol_Total": chol_tot,
        "Cholesterol_HDL": chol_hdl,
        "Cholesterol_LDL": chol_ldl,
        "GGT": ggt,
        "Serum_Urate": urate,
        "Physical_Activity_Level": activity,
        "Dietary_Intake_Calories": calories,
        "Alcohol_Consumption": None if alcohol == "None/Unknown" else alcohol,
        "Smoking_Status": smoking,
        "Family_History_of_Diabetes": fam_hist,
        "Previous_Gestational_Diabetes": gest_diab,
    }])

    clf_cols = clf_bundle["feature_columns"]
    reg_cols = reg_bundle["feature_columns"]

    clf_pred = int(clf_bundle["pipeline"].predict(row.reindex(columns=clf_cols))[0])
    reg_pred = float(reg_bundle["pipeline"].predict(row.reindex(columns=reg_cols))[0])

    st.success(f"Classification (High Risk): {'Yes (1)' if clf_pred == 1 else 'No (0)'}")
    st.info(f"Regression (Predicted HbA1c): {reg_pred:.2f}")