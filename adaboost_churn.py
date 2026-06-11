import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the best model and the scaler
try:
    model = joblib.load('adaboost_best_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'adaboost_best_model.joblib' and 'standard_scaler.joblib' are in the same directory as this app.py.")
    st.stop()

st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn.')

# Define the input features based on the X DataFrame columns
feature_names = {feature_names}

# Create input widgets for each feature
input_data = {{}}

# Categorical features that were one-hot encoded
categorical_features = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Numeric features
numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

for feature in feature_names:
    if feature in numeric_features:
        if feature == 'SeniorCitizen':
            input_data[feature] = st.selectbox(f"{{feature.replace('_', ' ')}}", [0, 1])
        elif feature == 'tenure':
            input_data[feature] = st.slider(f"{{feature.replace('_', ' ')}}", 0, 72, 1)
        elif feature == 'MonthlyCharges':
            input_data[feature] = st.number_input(f"{{feature.replace('_', ' ')}}", min_value=0.0, max_value=120.0, value=50.0, step=0.1)
        elif feature == 'TotalCharges':
            input_data[feature] = st.number_input(f"{{feature.replace('_', ' ')}}", min_value=0.0, max_value=8684.8, value=1000.0, step=0.1)
    elif feature in categorical_features:
        # For one-hot encoded boolean features, use a checkbox or selectbox for the original category if needed.
        # Here, assuming they directly represent a boolean choice like 'Yes' or 'No internet service'
        # For simplicity, we'll represent them as boolean inputs directly.
        input_data[feature] = st.selectbox(f"{{feature.replace('_', ' ')}}", [False, True])

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the order of columns matches the training data
input_df = input_df[feature_names]

# Scale the input data
scaled_input = scaler.transform(input_df)

if st.button('Predict Churn'):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.write(f"## Prediction: Customer is likely to Churn")
    else:
        st.write(f"## Prediction: Customer is unlikely to Churn")
        
    st.write(f"Probability of Churn: {{prediction_proba[0][1]:.2f}}")
    st.write(f"Probability of No Churn: {{prediction_proba[0][0]:.2f}}")
