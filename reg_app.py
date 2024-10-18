import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


## Load trained model, scaler pickle, onehot pickle file

model = tf.keras.models.load_model('reg_model.h5')

## load the encoder and scaler

with open('reg_label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('reg_onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('reg_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit app

st.title = ('Customer Churn Prediction')

#User input

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
Exited = st.selectbox('Is Exited', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [Exited],
})

# One hot encode Geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))


#combine one-hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
#Scaling the input data

input_data_scaled = scaler.transform(input_data)

#Predict churn

prediction = model.predict(input_data_scaled)
predicted_salary = prediction [0] [0]
st.write(f'Predicted Salary:', round((predicted_salary),2), '€')

