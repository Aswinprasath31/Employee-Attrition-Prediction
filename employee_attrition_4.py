import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction - Enhanced", layout="wide")

APP_TITLE = "Employee Attrition Prediction - Enhanced"
MODEL_PATH = 'model/model.pkl'
SCALER_PATH = 'model/scaler.pkl'
AUTHOR = "Aswin Prasath V"

# --- Load model and scaler using pickle ---
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

st.title(APP_TITLE)
st.write(f"Developed by: **{AUTHOR}**")
st.write("""
Upload a CSV file with employee data (features same as training data except 'Attrition'), 
or test a single employee below.
""")

def convert_label(p):
    if p == 'Yes':
        return 1
    elif p == 'No':
        return 0
    else:
        try:
            return int(p)
        except ValueError:
            return 0

# --- Sidebar for single employee input ---
st.sidebar.header("Single Employee Prediction")

def single_employee_input():
    BusinessTravel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    Department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    EducationField = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    JobRole = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    OverTime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    
    Age = st.sidebar.number_input("Age", 18, 65, 30)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    
    emp_data = {
        'Age': Age,
        'MonthlyIncome': MonthlyIncome,
        'BusinessTravel': BusinessTravel,
        'Department': Department,
        'EducationField': EducationField,
        'Gender': Gender,
        'JobRole': JobRole,
        'MaritalStatus': MaritalStatus,
        'OverTime': OverTime
    }
    return pd.DataFrame([emp_data])

# --- Main section ---
mode = st.radio("Select mode", ["Batch CSV Upload", "Single Employee Prediction"])

if mode == "Single Employee Prediction":
    st.subheader("Single Employee Attrition Prediction")
    single_df = single_employee_input()
    st.dataframe(single_df)

    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    single_encoded = pd.get_dummies(single_df, columns=categorical_cols, drop_first=True)

    expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else model.feature_names_in_

    for col in expected_cols:
        if col not in single_encoded.columns:
            single_encoded[col] = 0
    single_encoded = single_encoded[expected_cols]

    single_scaled = scaler.transform(single_encoded)

    pred_prob = model.predict_proba(single_scaled)[:, 1][0]
    pred_label = model.predict(single_scaled)[0]

    st.write(f"**Attrition Probability:** {pred_prob:.2f}")
    st.write(f"**Prediction:** {'Yes' if pred_label == 1 else 'No'}")

else:
    st.subheader("Batch CSV Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else model.feature_names_in_
        for col in expected_cols:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

        data_encoded = data_encoded[expected_cols]
        data_scaled = scaler.transform(data_encoded)

        predictions = model.predict(data_scaled)
        probs = model.predict_proba(data_scaled)[:, 1]

        data['Attrition_Predicted'] = predictions
        data['Attrition_Probability'] = probs

        st.write("Predictions:")
        st.dataframe(data[['Attrition_Predicted', 'Attrition_Probability']])

        st.write("Probability Distribution:")
        fig, ax = plt.subplots()
        ax.hist(probs, bins=20, color='skyblue')
        st.pyplot(fig)

        threshold = st.slider("Filter by Probability", 0.0, 1.0, 0.5)
        filtered = data[data['Attrition_Probability'] >= threshold]
        st.write(f"Filtered Employees (probability ≥ {threshold}):")
        st.dataframe(filtered)

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered CSV", csv, "filtered_attrition.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown(f"Developed by {AUTHOR} | Powered by Streamlit + scikit-learn")
