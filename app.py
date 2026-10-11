import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
page_title="Diabetes Classification",
page_icon="🩺"
)

@st.cache_resource
def load_model():
  return joblib.load("diabetes_model.pk1")

model = load_model()

st.title("Diabetes Status Classification")
st.write(
"Enter the information below to see the model's classification."
)

age = st.number_input("Age (years)", min_value=20, max_value=120, value=35)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=350.0, value=75.0)
hba1c = st.number_input("HbA1c (%)", min_value=2.0, max_value=20.0, value=5.5)
cholesterol = st.number_input(
"Total cholesterol (mg/dL)",
min_value=50.0, max_value=600.0, value=180.0
)

family_history = st.radio(
"Family history of diabetes?",
["No", "Yes"]
)

if st.button("Classify"):
  bmi = weight / ((height / 100) ** 2)

input_data = pd.DataFrame([{
  "RIDAGEYR": age,
  "BMXHT": height,
  "BMXWT": weight,
  "BMXBMI": bmi,
  "LBXGH": hba1c,
  "LBXTC": cholesterol,
  "family_history": 1 if family_history == "Yes" else 0
}])

prediction = int(model.predict(input_data)[0])
probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
  st.warning("The model predicts: Recorded diagnosis class")
else:
  st.success("The model predicts: No recorded diagnosis class")

st.write(f"Model probability for the recorded-diagnosis class: {probability:.1%}")

st.caption(
"Educational demonstration only. This result is not a medical diagnosis."
)
