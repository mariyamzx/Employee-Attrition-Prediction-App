# 🏢 Employee Attrition Prediction App  

An interactive *Streamlit* web app that predicts whether an employee is likely to leave the company — powered by a *custom Logistic Regression model* that outperformed sklearn’s implementation.  

---

## 🚀 Features  
- *Custom Logistic Regression: Implemented from scratch using the **Sigmoid* function and *Gradient Descent*.  
- *Data Preprocessing Pipeline*: Includes outlier capping, encoding categorical variables, and feature scaling.  
- *Feature Engineering*: Selection of the top contributing features to improve model interpretability and performance.  
- *EDA (Exploratory Data Analysis)*: Distribution analysis, correlation checks, and identification of key attrition drivers.  
- *Interactive Predictions*: Input employee details in the app and instantly see the attrition prediction — with visuals & GIFs.  
- *Clean, Intuitive UI*: Built with Streamlit, styled for a professional look (including dark mode compatibility).  

---

## 📊 Dataset  
- Sourced from an Employee Attrition dataset (IBM HR Analytics-style).  
- Contains employee demographic, role, and performance data.  
- Target variable: *Attrition* (Yes / No).  

---

## ⚙ Tech Stack  
- *Python* 🐍
- *Matplotlib and seaborn* for eda
- *Streamlit* for web interface  
- *Scikit-learn* (for preprocessing tools)  
- *Pandas & NumPy* for data handling  
- *Custom ML Implementation: Logistic Regression from scratch  (using **gradient descent* and *sigmoid function*)

---

## 🛠 How It Works  

### 1️⃣ Data Preprocessing  
- Missing value handling  
- Outlier treatment using a **custom Capping transformer** (IQR-based capping)  
- Categorical encoding (One-Hot Encoding)  
- Feature scaling  

### 2️⃣ Model Training  
- *Custom Logistic Regression* class implemented with:  
  - Sigmoid function  
  - Batch Gradient Descent optimization  
- Compared with sklearn’s LogisticRegression — *custom version showed higher accuracy* on test data.  

### 3️⃣ Deployment  
- Entire preprocessing + model pipeline serialized with pickle.  
- Streamlit app loads the pipeline and accepts user inputs for prediction.  

---

## 📦 Installation  


### Clone the repository
git clone https://github.com/mariyamzx/Employee-Attrition-Prediction-App.git
cd Employee-Attrition-Prediction-App

### Install dependencies
pip install -r requirements.txt

## ▶ Usage

streamlit run app/main.py

## 📸 App Preview
(You can insert a screenshot or GIF preview here)

## 🔮 Future Improvements
Add more model options (Decision Trees, Random Forest) for comparison.

Improve UI with more visual analytics.

Deploy online via Streamlit Cloud or Hugging Face Spaces.

## 👨‍💻 Author
Developed by *@mariyamzx*