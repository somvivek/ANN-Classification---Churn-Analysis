import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# load the trained model, scaler, and data
model = load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Customer Churn Prediction')
st.write('This app predicts customer churn based on user input.')

# Input fields
creditscore = st.number_input('Credit Score', min_value=300, max_value=850, value=700)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_credit_card = st.selectbox('Has Credit Card', ['1', '0'])
is_active_member = st.selectbox('Is Active Member', ['1', '0'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=150000.0, value=50000.0)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode categorical features
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_data.drop(['Geography'], axis=1, inplace=True)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
predictions = model.predict(input_data_scaled)
predictions_prob = predictions[0][0]
predictions_prob

if predictions_prob > 0.5:
    st.write(f"Customer is likely to churn with a probability of {predictions_prob:.2f}")
else:
    st.write(f"Customer is likely to stay with a probability of {1 - predictions_prob:.2f}")