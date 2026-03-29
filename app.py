import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Approval ML App", layout="wide")

# ===============================
# Upload Dataset
# ===============================
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)
    return df

# Default fallback
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = pd.read_csv("train.csv")

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA Analysis", "Visualizations", "Prediction"]
)

# ===============================
# PAGE 1 — Dataset Overview
# ===============================
if page == "Dataset Overview":
    st.title("📊 Loan Prediction Project")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ===============================
# PAGE 2 — EDA
# ===============================
elif page == "EDA Analysis":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values Heatmap")

    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(df.isnull(), yticklabels=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # Handle missing values
    st.subheader("Handle Missing Values")
    if st.button("Fill Missing Values"):
        df.fillna(df.mode().iloc[0], inplace=True)
        st.success("Missing values filled using mode")

# ===============================
# PAGE 3 — Visualizations
# ===============================
elif page == "Visualizations":
    st.title("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="Loan_Status", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===============================
# PAGE 4 — Prediction (IMPROVED)
# ===============================
elif page == "Prediction":
    st.title("🤖 Loan Approval Prediction (Advanced AI System)")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Term", min_value=1)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict"):

        TotalIncome = ApplicantIncome + CoapplicantIncome
        EMI = LoanAmount / Loan_Amount_Term

        input_data = pd.DataFrame({
            "Gender":[Gender],
            "Married":[Married],
            "Dependents":[Dependents],
            "Education":[Education],
            "Self_Employed":[Self_Employed],
            "ApplicantIncome":[ApplicantIncome],
            "CoapplicantIncome":[CoapplicantIncome],
            "LoanAmount":[LoanAmount],
            "Loan_Amount_Term":[Loan_Amount_Term],
            "Credit_History":[Credit_History],
            "Property_Area":[Property_Area],
            "TotalIncome":[TotalIncome],
            "EMI":[EMI]
        })

        # Encoding
        for col in input_data.select_dtypes(include="object").columns:
            input_data[col] = input_data[col].astype("category").cat.codes

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # ===============================
        # RESULT
        # ===============================
        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {1-prob:.2f})")

        # ===============================
        # 🔥 NEW FEATURE 1: RISK SCORE
        # ===============================
        risk_score = int((1 - prob) * 100)
        st.subheader(f"📊 Risk Score: {risk_score}/100")

        # ===============================
        # 🔥 NEW FEATURE 2: FRAUD DETECTION
        # ===============================
        if ApplicantIncome < 2000 and LoanAmount > 500:
            st.warning("⚠️ Possible Fraud Detected (Low income vs high loan)")

        # ===============================
        # 🔥 NEW FEATURE 3: EXPLANATION
        # ===============================
        st.subheader("🧠 AI Explanation")

        if Credit_History == 1.0:
            st.write("✔️ Good credit history increases approval chances")
        else:
            st.write("❌ Poor credit history reduces approval chances")

        if TotalIncome > 5000:
            st.write("✔️ High income supports loan approval")
        else:
            st.write("❌ Low income reduces approval chances")

        if EMI > (TotalIncome * 0.4):
            st.write("❌ EMI too high compared to income")

        # ===============================
        # 🔥 NEW FEATURE 4: SMART SUGGESTION
        # ===============================
        st.subheader("💡 Improvement Suggestions")

        if prediction == 0:
            st.write("- Increase income or add co-applicant")
            st.write("- Reduce loan amount")
            st.write("- Improve credit score")
            st.write("- Choose longer loan term")

       
        # ===============================
        # DOWNLOAD
        # ===============================
        input_data["Prediction"] = prediction
        st.download_button(
            "Download Result",
            input_data.to_csv(index=False),
            "prediction.csv",
            "text/csv"
        )
