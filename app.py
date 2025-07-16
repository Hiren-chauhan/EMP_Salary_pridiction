import streamlit as st
import pandas as pd
import numpy as np
from main import SalaryPredictor
# Load the dataset
df = pd.read_csv("adult_3.csv")
# Create and train the model
predictor = SalaryPredictor()
predictor.fit(df.copy())  # Pass a copy to avoid modifying the original df
# Streamlit app
st.set_page_config(
    page_title="Employee Salary Prediction [AICTE B2_AI - (2025-26) ]", layout="wide"
)
st.title("Employee Salary Prediction [AICTE B2_AI - (2025-26) ]")
st.write(
    "This application predicts whether an employee's income is greater than $50K based on their demographic data. The model has an accuracy of {:.2f}%".format(
        predictor.accuracy * 100
    )
)
# Sidebar for user input
st.sidebar.header("Enter Employee Information")
# Get user input for prediction
age = st.sidebar.slider("Age", 17, 90, 30)
workclass = st.sidebar.selectbox("Work Class", df["workclass"].unique())
educational_num = st.sidebar.slider("Education Level (Numeric)", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", df["marital-status"].unique())
occupation = st.sidebar.selectbox("Occupation", df["occupation"].unique())
relationship = st.sidebar.selectbox("Relationship", df["relationship"].unique())
race = st.sidebar.selectbox("Race", df["race"].unique())
gender = st.sidebar.selectbox("Gender", df["gender"].unique())
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", df["native-country"].unique())
# Create a dataframe from user input
user_input = pd.DataFrame(
    {
        "age": [age],
        "workclass": [workclass],
        "educational-num": [educational_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
    }
)
# Add all other columns from the original dataframe that are not in the user input,
for col in predictor.X_columns:
    if col not in user_input.columns:
        user_input[col] = df[col].mode()[0]  # Use mode for missing categorical features
# Make prediction
prediction, prediction_proba = predictor.predict(user_input)
# Display prediction
st.subheader("Prediction Result")
if prediction[0] == " >50K":
    st.success(f"**Predicted Income:** {prediction[0]}")
else:
    st.info(f"**Predicted Income:** {prediction[0]}")
st.write(f"**Confidence:** {np.max(prediction_proba) * 100:.2f}%")
st.subheader("Data Preview")
st.write(df.head(10))
