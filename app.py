import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from PIL import Image

# Load your LightGBM model
model = lgb.Booster(model_file='model_lgbm.txt')

st.title('Customer Churn Prediction')
st.sidebar.header('Customer Details')
# image = Image.open('customers.jpg')
# st.(caption='Predict Customer Churn')

# FUNCTION
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    Partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    Dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 1)
    PhoneService = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    StreamingTV = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.25, 118.75, 25.0)
    TotalCharges = st.sidebar.slider('Total Charges', 18.8, 8684.8, 100.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.header('Customer Details for Prediction')
st.write(input_df)

# Preprocess input data here as required before prediction
# For example, encode categorical variables if your model requires that

# Prediction
# Ensure your model's predict method matches this call
prediction_proba = model.predict(input_df)

st.subheader('Churn Probability')
st.write(f'Probability of Churn: {prediction_proba[0]:.2f}')
