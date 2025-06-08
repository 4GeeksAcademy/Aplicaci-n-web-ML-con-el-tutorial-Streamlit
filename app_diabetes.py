import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Archivos
MODEL_PATH = "modelo_diabetes.pkl"
DATA_PATH = "diabetes.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def train_and_save_model(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        data = load_data()
        return train_and_save_model(data)

st.title("Predicción de Diabetes")

def user_input():
    return pd.DataFrame([{
        "Pregnancies": st.number_input("Embarazos", 0),
        "Glucose": st.slider("Glucosa", 0, 200, 120),
        "BloodPressure": st.slider("Presión sanguínea", 0, 140, 70),
        "SkinThickness": st.slider("Espesor de piel", 0, 100, 20),
        "Insulin": st.slider("Insulina", 0, 900, 80),
        "BMI": st.slider("IMC", 0.0, 70.0, 25.0),
        "DiabetesPedigreeFunction": st.slider("DPF", 0.0, 3.0, 0.5),
        "Age": st.slider("Edad", 10, 100, 33)
    }])

input_df = user_input()
model = load_or_train_model()

if st.button("Predecir"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error(f"El modelo predice que el paciente **tiene diabetes**. (Probabilidad: {prob:.2f})")
    else:
        st.success(f"El modelo predice que el paciente **no tiene diabetes**. (Probabilidad: {prob:.2f})")

        