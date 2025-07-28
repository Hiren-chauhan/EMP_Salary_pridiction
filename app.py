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
    page_title="Employee Salary Prediction [AICTE B2_AI - (2025-26) ]",
    layout="wide",
    initial_sidebar_state="expanded",
)
# --- UI Styling ---
st.markdown(
    """
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .st-emotion-cache-1g6gooi {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)
# --- Header ---
st.title("📊 Employee Salary Prediction [AICTE B2_AI - (2025-26) ] By Hiren Chauhan")
st.write(
    "This app predicts whether an employee's income exceeds $50K based on their demographic data."
)
st.write(f"**🤖 Model Accuracy:** {predictor.accuracy * 100:.2f}%")
# --- Sidebar for User Input ---
st.sidebar.header("👤 Enter Employee Information")
age = st.sidebar.slider("Age", 17, 65, 30, help="Select the employee's age.")
workclass = st.sidebar.selectbox(
    "Work Class", df["workclass"].unique(), help="Select the type of employment."
)
educational_num = st.sidebar.slider(
    "Education Level (Numeric)",
    1,
    16,
    10,
    help="Select the employee's education level on a numeric scale.",
)
occupation = st.sidebar.selectbox(
    "Occupation", df["occupation"].unique(), help="Select the employee's occupation."
)
hours_per_week = st.sidebar.slider(
    "Hours per Week", 1, 99, 40, help="Enter the number of hours worked per week."
)
native_country = st.sidebar.selectbox(
    "Native Country",
    df["native-country"].unique(),
    help="Select the employee's native country.",
)
# --- Footer in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🧑‍💻:** Hiren Chauhan")
st.sidebar.markdown("**📧:** Hinesh13@gmail.com")
st.sidebar.markdown("**🎓:** [AICTE B2_AI - (2025-26) ]")
# --- Main Panel for Displaying Results ---
# Create a DataFrame from user's sidebar selections for display
display_input = pd.DataFrame(
    {
        "Feature": [
            "Age",
            "Work Class",
            "Education Level",
            "Occupation",
            "Hours per Week",
            "Native Country",
        ],
        "Selected Value": [
            age,
            workclass,
            educational_num,
            occupation,
            hours_per_week,
            native_country,
        ],
    }
)
# Create the full input DataFrame for the model, filling missing values with the mode
model_input = pd.DataFrame(
    {
        "age": [age],
        "workclass": [workclass],
        "educational-num": [educational_num],
        "occupation": [occupation],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
    }
)
# Fill missing columns with the mode from the training data
for col in predictor.X_columns:
    if col not in model_input.columns:
        model_input[col] = df[col].mode()[0]
# Make prediction
prediction, prediction_proba = predictor.predict(model_input)
confidence = np.max(prediction_proba) * 100
# --- Layout for Inputs and Prediction ---
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.subheader("📋 Your Selections")
    st.table(display_input.set_index("Feature"))
with col2:
    st.subheader("💡 Prediction Result")
    if prediction[0] == " >50K":
        st.success("High Income Predicted! 🥳")
        st.metric(label="Predicted Salary", value="> $50K")
    else:
        st.info("Standard Income Predicted. 👍")
        st.metric(label="Predicted Salary", value="<= $50K")
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
# --- Data Preview ---
with st.expander("Show Data Preview"):
    st.write("Here is a small preview of the dataset used for training the model:")
    st.dataframe(df.head(10))
