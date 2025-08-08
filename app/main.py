import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


# ======== Custom Classes for Unpickling ========
class Capping(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_capped = X.copy()
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            iqr = Q3 - Q1
            lower = Q1 - 1.5 * iqr
            upper = Q3 + 1.5 * iqr
            X_capped[col] = np.where(X_capped[col] < lower, lower,
                              np.where(X_capped[col] > upper, upper, X_capped[col]))
        return X_capped

    def get_feature_names_out(self, input_features=None):
        return input_features

class LogisticGD:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.coef_ = np.ones(X.shape[1])
        for _ in range(self.epochs):
            y_hat = self.sigmoid(np.dot(X, self.coef_))
            self.coef_ += self.lr * np.dot(y - y_hat, X) / X.shape[0]
        return self
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.coef_)
        y_hat = self.sigmoid(z)
        return np.where(y_hat >= 0.5, 1, 0)

# ======== Streamlit Config ========
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢", layout="wide")

# ======== Load Pipeline ========
@st.cache_data
def load_pipeline():
    with open("Attrition.pkl", "rb") as file:
        return pickle.load(file)

pipeline = load_pipeline()

def add_sidebar():
    st.sidebar.header("📋 Employee Information")

    user_data = {
        "OverTime": st.sidebar.selectbox("OverTime", ["Yes", "No"]),
        "JobRole": st.sidebar.selectbox("JobRole", [
            "Laboratory Technician", "Human Resources", "Sales Representative",
            "Manager", "Research Director", "Manufacturing Director"
        ]),
        "Department": st.sidebar.selectbox("Department", [
            "Research & Development", "Sales", "Human Resources"
        ]),
        "EducationField": st.sidebar.selectbox("Education Field", [
            "Technical Degree", "Marketing"
        ]),
        "YearsSinceLastPromotion": st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=1),
        "MonthlyRate": st.sidebar.number_input("Monthly Rate", min_value=1000, max_value=30000, value=10000),
        "DailyRate": st.sidebar.number_input("Daily Rate", min_value=100, max_value=2000, value=800),
        "StockOptionLevel": st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3]),
        "Age": st.sidebar.number_input("Age", min_value=18, max_value=65, value=30),
        "MonthlyIncome": st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000),

        # Defaults for missing features
        "TrainingTimesLastYear": 3,
        "BusinessTravel": "Travel_Rarely",
        "YearsWithCurrManager": 5,
        "PercentSalaryHike": 15,
        "MaritalStatus": "Single",
        "TotalWorkingYears": 10,
        "YearsInCurrentRole": 4,
        "DistanceFromHome": 5,
        "HourlyRate": 60,
        "Gender": "Male",
        "YearsAtCompany": 6,
        "NumCompaniesWorked": 3
    }

    return pd.DataFrame([user_data])

# ======== Main App ========
def main():
    st.title("🏢 Employee Attrition Prediction")
    # st.write("### Predict whether an employee is likely to leave the company based on key factors.")
    st.markdown("<h5 style='color:gray;'>Predict whether an employee is likely to leave the company based on key factors such as role, salary, work experience, and personal details. This tool uses machine learning to provide insights and help HR teams take proactive steps.</h5>", unsafe_allow_html=True)

    input_df = add_sidebar()

    st.subheader("Employee Profile")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Attrition", use_container_width=True):
         prediction = pipeline.predict(input_df)[0]
         col1, col2 = st.columns([1, 1])
         if prediction == 1:
            with col1:
                st.error("🚨 Likely to Leave")
            with col2:
                st.image("https://media.giphy.com/media/KDRv3QggAjyo/giphy.gif", width=200)
         else:
            with col1:
                st.success("✅ Likely to Stay")
            with col2:
                st.image("https://media.giphy.com/media/Ge86XF8AVY1KE/giphy.gif", width=200)

if __name__ == "__main__":
    main()
