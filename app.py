# ==========================================================
# Customer Churn Prediction Model
# Replace your existing app.py with this file and keep
# customer_churn_model.pkl and encoder.pkl in the same folder.
# ==========================================================

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction Model",
    layout="wide"
)

# ---------- Load model ----------
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ---------- CSS ----------
st.markdown("""
<style>
.stApp{
    background:#f4f7fb;
}
.block-container{
    padding-top:1.5rem;
    padding-bottom:2rem;
}
div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:16px;
    box-shadow:0 2px 10px rgba(0,0,0,.08);
}
.stButton>button{
    width:100%;
    border-radius:14px;
    height:52px;
    background:#2563eb;
    color:white;
    font-weight:700;
    font-size:18px;
}
.stButton>button:hover{
    background:#1d4ed8;
}
section[data-testid="stSidebar"]{
    background:#eef4ff;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div style="background:linear-gradient(90deg,#2563EB,#1E40AF);
padding:30px;border-radius:18px;color:white;text-align:center;">
<h1> Customer Churn Prediction Model</h1>
<p>Predict customer churn and receive business recommendations.</p>
</div>
""", unsafe_allow_html=True)


st.subheader("Customer Information")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### 👤 Customer")
        gender = st.selectbox("Gender",["Male","Female"])
        Partner = st.selectbox("Partner",["Yes","No"])
        Dependents = st.selectbox("Dependents",["Yes","No"])
        SeniorCitizen = st.selectbox("Senior Citizen",[0,1])
        tenure = st.slider("Tenure",0,72,12)

with c2:
    with st.container(border=True):
        st.markdown("### 📡 Services")
        PhoneService = st.selectbox("Phone Service",["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
        InternetService = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("Online Security",["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("Online Backup",["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])
        TechSupport = st.selectbox("Tech Support",["Yes","No","No internet service"])
        StreamingTV = st.selectbox("Streaming TV",["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])

with c3:
    with st.container(border=True):
        st.markdown("### 💳 Billing")
        Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method",[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        MonthlyCharges = st.slider("Monthly Charges",18.0,120.0,70.0)
        TotalCharges = st.number_input("Total Charges",value=1000.0)

categorical_df = pd.DataFrame({
    "gender":[gender],
    "Partner":[Partner],
    "Dependents":[Dependents],
    "PhoneService":[PhoneService],
    "MultipleLines":[MultipleLines],
    "InternetService":[InternetService],
    "OnlineSecurity":[OnlineSecurity],
    "OnlineBackup":[OnlineBackup],
    "DeviceProtection":[DeviceProtection],
    "TechSupport":[TechSupport],
    "StreamingTV":[StreamingTV],
    "StreamingMovies":[StreamingMovies],
    "Contract":[Contract],
    "PaperlessBilling":[PaperlessBilling],
    "PaymentMethod":[PaymentMethod]
})

encoded = encoder.transform(categorical_df)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

numeric_df = pd.DataFrame({
    "SeniorCitizen":[SeniorCitizen],
    "tenure":[tenure],
    "MonthlyCharges":[MonthlyCharges],
    "TotalCharges":[TotalCharges]
})

final_df = pd.concat([numeric_df, encoded_df], axis=1)

st.divider()

if st.button(" Predict Churn", use_container_width=True):
    pred = model.predict(final_df)[0]
    probs = model.predict_proba(final_df)[0]
    confidence = probs.max()

    st.subheader(" Prediction Results")
    prediction = model.predict(final_df)

    probability = model.predict_proba(final_df)
    st.write("Confidence:", probability.max())

    st.progress(float(confidence))

    with st.container(border=True):
        st.subheader(" Business Recommendations")
        if pred:
            st.error(" High Churn Risk")
            for rec in [
                " Offer promotional discounts",
                " Contact the customer",
                "Recommend a long-term contract",
                "Prioritize technical support",
                "Offer loyalty rewards",
            ]:
                st.write(rec)
        else:
            st.success(" Low Churn Risk")
            for rec in [
                " Maintain service quality",
                " Recommend premium plans",
                " Offer referral bonuses",
                " Collect customer feedback",
                " Reward loyal customers",
            ]:
                st.write(rec)

with st.expander(" Model Information"):
    st.write("""
- Binary Classification
- Customer Churn Prediction
- OneHotEncoder + Scikit-learn Model
- Built with Streamlit
""")

