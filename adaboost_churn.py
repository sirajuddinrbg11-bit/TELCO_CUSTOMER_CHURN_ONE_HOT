import streamlit as st
import pandas as pd
import pickle 
import joblib
adaboost_best_model = joblib.load('adaboost_best_model.pkl')
columns=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn',
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
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Tenure = st.slider("Tenure", min_value=None, 0,72,1)



Input_data=pd.DataFrame([[SeniorCitizen,Tenure]],columns=columns)
if st.button("Predict_Customer_Churn):
  prediction=predict_customer_churn(Input_data)
  if prediction[0]==1:
    st.write("Cutomer is likely to churn")
  else:
    st.write("Customer is unlikely to churn")
