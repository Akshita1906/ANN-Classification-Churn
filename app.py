import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model("ann_model.h5")

# Load the label encoders and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("one_hot_encoder_geo.pkl", "rb") as f:
    lable_encoder_geo = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox("Geography", lable_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,100)
tenure = st.slider("Tenure",0,10)
balance = st.number_input("Balance",0,250000)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")
creditscore = st.number_input("Credit Score")

input_data = pd.DataFrame({
    "CreditScore": [creditscore],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geography_encoded = lable_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geography_encoded, columns=lable_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

#scale the input data
input_scaled = scaler.transform(input_data)

#predict churn probability
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_proba:.2f}")
