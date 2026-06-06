import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

import joblib
# Download and load the trained model
model_path = hf_hub_download(repo_id="toriaiml/Vehicles-Predictive-Maintenance", filename="predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Vehicle Predictive Maintenance")
st.write("""
This application predicts whether the engine requires maintenance or is operating normally.

Please enter the engine sensor values below to predict whether maintenance is required
""")

# User input

EngineRPM = st.number_input("Engine RPM", min_value=10, max_value=3000)
LubOilPressure = st.number_input("Lubricant Oil pressure", min_value=0.0, max_value=10.0)
FuelPressure = st.number_input("Fuel Pressure", min_value=0.0, max_value=20.0)
CoolantPressure = st.number_input("Coolant Pressure", min_value=0.0, max_value=9.0)
LubOilTemp = st.number_input("Lubricant Oil Temperature", min_value=70.0, max_value=90.0)
CoolantTemp = st.number_input("Coolant Temperature", min_value=50.0, max_value=120.0)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': EngineRPM,
    'Lub oil pressure': LubOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LubOilTemp,
    'Coolant temp': CoolantTemp
}])

# Predict button
if st.button("Predict Engine Condition"):
    maintenance_prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if maintenance_prob >= 0.45 else 0
    if prediction == 1:
       st.error("Maintenance Required")
    else:
        st.success("Engine Operating Normally")
