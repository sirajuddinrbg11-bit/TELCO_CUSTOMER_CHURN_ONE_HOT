import streamlit as st
import pandas as pd
import pickle 
import joblib
adaboost_best_model = joblib.load('adaboost_best_model.pkl')
columns=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
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
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
def predict_customer_churn(features):
  prediction=adaboost_best_model.predict(features)
  return prediction


st.title("Customer Churn Prediction")

# get user Input
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

Input_data=pd.DataFrame([[SeniorCitizen,tenure,MonthlyCharges,TotalCharges]],columns=columns)
if st.button("Predict_Customer_Churn):
  prediction=predict_customer_churn(Input_data)
  if prediction[0]==1:
    st.write("Cutomer is likely to churn")
  else:
    st.write("Customer is unlikely to churn")
