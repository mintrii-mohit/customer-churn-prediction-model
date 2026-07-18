# 📉 Customer Churn Prediction

A Machine Learning web application that predicts whether a customer is likely to churn based on customer demographics, account information, and subscribed services. The project helps businesses identify high-risk customers and take proactive retention measures.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones. This project builds a predictive machine learning model that classifies whether a customer is likely to leave the company.

The project covers the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 🎯 Objectives

- Predict customer churn using historical customer data.
- Identify customers at high risk of leaving.
- Help businesses improve customer retention strategies.
- Deploy the trained model as an interactive web application.

---

## 📂 Dataset

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Multiple Lines
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Churn (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Render (Deployment)

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection
- Imported customer churn dataset.

### 2. Data Preprocessing
- Removed unnecessary columns.
- Handled missing values.
- Encoded categorical features.
- Prepared data for model training.

### 3. Exploratory Data Analysis (EDA)
- Analyzed customer demographics.
- Studied churn distribution.
- Identified important relationships between features.

### 4. Model Training
Different classification algorithms were tested, and the best-performing model was selected.

### 5. Model Evaluation
The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 6. Deployment
The trained model was deployed using Streamlit, allowing users to enter customer details and receive instant churn predictions.

---

## 📊 Features

- Interactive Streamlit interface
- Real-time churn prediction
- User-friendly input forms
- Fast prediction results
- Deployment-ready application

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained ML model
├── churn.csv               # Dataset
├── requirements.txt        # Required libraries
├── README.md               # Project documentation
└── assets/                 # Images (optional)
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Hyperparameter tuning
- Feature importance visualization
- SHAP explainability
- Multiple ML model comparison
- Cloud deployment using Docker
- Database integration

---

## 💡 Business Impact

This project enables businesses to:

- Detect customers likely to churn.
- Improve customer retention.
- Reduce revenue loss.
- Make data-driven business decisions.
- Optimize customer engagement strategies.

---

## 👨‍💻 Author

**Mohit Mintri**

B.Tech Chemical Engineering, NIT Agartala

Aspiring Data Analyst | Machine Learning Enthusiast

---

## ⭐ If you found this project useful, don't forget to Star the repository!
