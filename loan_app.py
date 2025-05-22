import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

# Load model và encoder đã huấn luyện sẵn
model = joblib.load("xgb_model.pkl")  # model đã được train
encoder = joblib.load("encoder.pkl")  # encoder đã được fit

# Các cột đặc trưng
num_features = ['CreditScore', 'MonthlyIncome', 'NetWorth', 'DebtToIncomeRatio', 
                'UtilityBillsPaymentHistory', 'MonthlyDebtPayments', 'Experience',
                'NumberOfDependents', 'LoanAmount', 'LoanDuration']

cat_features = ['LoanPurpose', 'MaritalStatus', 'EmploymentStatus', 'EducationLevel', 'HomeOwnershipStatus']

# Hàm gợi ý
def suggest_top1_per_reduction(applicant, model, encoder, num_features, cat_features,
                               reduction_pcts=[10, 20, 30, 50], max_add_months=60):
    base_loan = applicant["LoanAmount"]
    base_duration = applicant["LoanDuration"]

    top_suggestions = []

    for pct in reduction_pcts:
        new_loan = base_loan * (1 - pct / 100)
        best_option = None
        best_proba = -1

        for add_month in range(1, max_add_months + 1):
            modified = applicant.copy()
            modified["LoanAmount"] = new_loan
            modified["LoanDuration"] = base_duration + add_month

            temp_df = pd.DataFrame([modified])
            temp_num = temp_df[num_features]
            temp_cat = pd.DataFrame(
                encoder.transform(temp_df[cat_features]),
                columns=encoder.get_feature_names_out(cat_features)
            )
            temp_processed = pd.concat([temp_num, temp_cat], axis=1)

            proba = model.predict_proba(temp_processed)[0, 1]

            if proba > best_proba:
                best_proba = proba
                best_option = {
                    "loan_pct": pct,
                    "new_amount": round(new_loan, 1),
                    "add_months": add_month,
                    "new_duration": modified["LoanDuration"],
                    "proba": round(proba * 100, 2)
                }

        if best_option:
            top_suggestions.append(best_option)

    return top_suggestions


# Giao diện Streamlit
st.title("Loan Approval Prediction App")

st.sidebar.header("Nhập thông tin hồ sơ")

# Nhập liệu
applicant = {
    "CreditScore": st.sidebar.number_input("Credit Score", 300, 850, 650),
    "MonthlyIncome": st.sidebar.number_input("Monthly Income", 0, 100000, 4000),
    "NetWorth": st.sidebar.number_input("Net Worth", 0, 1000000, 70000),
    "DebtToIncomeRatio": st.sidebar.slider("Debt to Income Ratio", 0.0, 1.0, 0.5),
    "UtilityBillsPaymentHistory": st.sidebar.slider("Utility Bills Payment History", 0.0, 1.0, 0.75),
    "MonthlyDebtPayments": st.sidebar.number_input("Monthly Debt Payments", 0, 100000, 800),
    "LoanAmount": st.sidebar.number_input("Loan Amount", 0, 1000000, 25000),
    "LoanDuration": st.sidebar.number_input("Loan Duration (months)", 1, 360, 24),
    "Experience": st.sidebar.number_input("Experience (years)", 0, 70, 5),
    "NumberOfDependents": st.sidebar.slider("Number of Dependents", 0, 10, 2),
    "MaritalStatus": st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"]),
    "EmploymentStatus": st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"]),
    "EducationLevel": st.sidebar.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"]),
    "HomeOwnershipStatus": st.sidebar.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage", "Other"]),
    "LoanPurpose": st.sidebar.selectbox("Loan Purpose", ["Auto", "Home", "Debt Consolidation", "Education", "Other"])
}

if st.button("Dự đoán khoản vay"):
    input_df = pd.DataFrame([applicant])
    input_num = input_df[num_features]
    input_cat = pd.DataFrame(
        encoder.transform(input_df[cat_features]),
        columns=encoder.get_feature_names_out(cat_features)
    )
    input_processed = pd.concat([input_num, input_cat], axis=1)

    probability = model.predict_proba(input_processed)[0, 1]
    prediction = 1 if probability > 0.75 else 0

    st.write(f"### Kết quả dự đoán: {'✅ Được duyệt' if prediction == 1 else '❌ Không được duyệt'}")
    st.write(f"Xác suất duyệt: **{round(probability * 100, 2)}%**")

    if prediction == 0:
        st.subheader("👉 Gợi ý các phương án thay thế:")
        suggestions = suggest_top1_per_reduction(applicant, model, encoder, num_features, cat_features)
        for s in suggestions:
            st.markdown(
                f"- 💰 **${s['new_amount']}** trong **{s['new_duration']} tháng** – "
                f"Xác suất chấp nhận: **{s['proba']}%**"
            )
