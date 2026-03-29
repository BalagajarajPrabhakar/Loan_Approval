import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Approval ML App", layout="wide")

# ===============================
# Load Data & Model
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = df.drop("Loan_ID", axis=1)
    return df

@st.cache_resource
def load_model():
    model = joblib.load("loan_model.pkl")
    return model

df = load_data()
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
    st.title("Loan Prediction Project")
    st.write("Dataset Shape:", df.shape)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())


# ===============================
# PAGE 2 — EDA
# ===============================
elif page == "EDA Analysis":
    st.title("Exploratory Data Analysis")

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Improved Missing Values Heatmap")

    fig, ax = plt.subplots(figsize=(14,6))

    sns.heatmap(
        df.isnull(),
        yticklabels=False,
        cbar=True,
        cmap="viridis",
        ax=ax
    )

    ax.set_title("Missing Values Distribution", fontsize=16)
    ax.set_xlabel("Features")
    ax.set_ylabel("Records")

    st.pyplot(fig)

    # Missing values count heatmap (extra useful)
    st.subheader("Missing Values Count")

    missing_data = df.isnull().sum().to_frame(name="Missing Count")

    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(
        missing_data,
        annot=True,
        fmt="d",
        cmap="magma",
        ax=ax2
    )

    st.pyplot(fig2)


# ===============================
# PAGE 3 — Visualizations
# ===============================
elif page == "Visualizations":
    st.title("Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Approval Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="Loan_Status", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Credit History vs Loan Status")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Correlation")

    # Only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)


# ===============================
# PAGE 4 — Prediction
# ===============================
elif page == "Prediction":
    st.title("Loan Approval Prediction")

    st.write("Enter applicant details:")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=1)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status"):

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

        # Encoding (same workflow preserved)
        for col in input_data.select_dtypes(include="object").columns:
            input_data[col] = input_data[col].astype("category").cat.codes

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("Loan Approved")
        else:

            st.error("Loan Rejected")
