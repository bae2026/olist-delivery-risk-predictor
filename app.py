
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn
model = joblib.load('app_delivery_lateness_model.pkl')
scaler = joblib.load('app_logistics_scaler.pkl')

#Setting up the UI
st.set_page_config(page_title="Olist Delivery Predictor", layout="centered")
st.title("🚚 Olist Logistics Risk Analyzer")

#Creating Input Fields
price = st.number_input("Product Price (BRL)", min_value=0.0, value=100.0)
freight = st.number_input("Freight Value (BRL)", min_value=0.0, value=20.0)
weight = st.number_input("Product Weight (grams)", min_value=0.0, value=1000.0)

# Prediction Logic
if st.button("Analyze Risk"):
    input_df = pd.DataFrame([[price, freight, weight]], 
                            columns=['price', 'freight_value', 'product_weight_g'])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    threshold = 0.3

    st.divider()


    if prediction[0] == 1:
        st.error(f"### ⚠️ HIGH RISK: Delayed Delivery Predicted")
        st.write(f"There is a **{probability:.1%}** probability of delay.")
    else:
        st.success(f"### ✅ LOW RISK: On-Time Delivery Predicted")
        st.write(f"There is a **{(1-probability):.1%}** confidence of on-time arrival.")
