import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set page config MUST be the very first Streamlit command
st.set_page_config(page_title="Employee Attrition Prediction - Enhanced", layout="wide")

# --- Your details ---
APP_TITLE = "Employee Attrition Prediction - Enhanced"
MODEL_PATH = 'employee_attrition_model.pkl'  # Update if needed
AUTHOR = "Aswin Prasath V"  # Your name

# --- Load model ---
@st.cache_resource  # cache model loading for better performance
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.title(APP_TITLE)
st.write(f"Developed by: **{AUTHOR}**")
st.write("""
Upload a CSV file with employee data (same features as training except 'Attrition').
Or try single employee prediction below.
""")

# Function to convert model predictions to int
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

# --- Sidebar: Single employee input ---
st.sidebar.header("Single Employee Attrition Prediction")

def single_employee_input():
    # Adjust input widgets as per your dataset's features
    BusinessTravel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    Department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    EducationField = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    JobRole = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    OverTime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    
    # Numeric inputs example - update with your actual numeric features
    Age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    
    # Build dict for this single record
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

# Main app: choose mode
mode = st.radio("Select mode", ["Batch CSV Upload", "Single Employee Prediction"])

if mode == "Single Employee Prediction":
    st.subheader("Predict Attrition for Single Employee")
    single_emp_df = single_employee_input()
    st.write("Employee Input Data:")
    st.dataframe(single_emp_df)
    
    # Preprocess single employee input same as batch (one-hot, missing cols)
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    single_encoded = pd.get_dummies(single_emp_df, columns=categorical_cols, drop_first=True)

    if hasattr(model.named_steps['scaler'], 'feature_names_in_'):
        expected_cols = model.named_steps['scaler'].feature_names_in_
    else:
        expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in single_encoded.columns:
            single_encoded[col] = 0
    single_encoded = single_encoded[expected_cols]

    # Predict single employee
    pred_prob = model.predict_proba(single_encoded)[:, 1][0]
    pred_label = model.predict(single_encoded)[0]
    pred_label_int = convert_label(pred_label)

    st.write(f"Predicted Attrition Probability: **{pred_prob:.2f}**")
    st.write(f"Predicted Attrition: **{'Yes' if pred_label_int == 1 else 'No'}**")

else:
    # Batch CSV upload mode
    st.subheader("Batch Prediction: Upload CSV file")
    uploaded_file = st.file_uploader("Upload your employee data CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Prepare data
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        if hasattr(model.named_steps['scaler'], 'feature_names_in_'):
            expected_cols = model.named_steps['scaler'].feature_names_in_
        else:
            expected_cols = model.feature_names_in_

        # Validate uploaded features
        missing_cols = set(expected_cols) - set(data_encoded.columns)
        if missing_cols:
            st.error(f"Uploaded data is missing expected columns: {missing_cols}")
            st.stop()

        # Add missing columns as zero
        for col in expected_cols:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

        data_encoded = data_encoded[expected_cols]

        # Predict
        pred_probs = model.predict_proba(data_encoded)[:, 1]
        predictions = model.predict(data_encoded)
        predictions = [convert_label(p) for p in predictions]

        # Add predictions to data
        data['Attrition_Predicted'] = predictions
        data['Attrition_Probability'] = pred_probs

        st.write("Predictions with Probability:")
        st.dataframe(data[['Attrition_Predicted', 'Attrition_Probability']])

        # Show histogram of probabilities
        st.write("Attrition Probability Distribution:")
        fig, ax = plt.subplots()
        ax.hist(pred_probs, bins=20, color='skyblue')
        st.pyplot(fig)

        # Filter by probability threshold
        threshold = st.slider("Filter employees by attrition probability threshold", 0.0, 1.0, 0.5)
        filtered_data = data[data['Attrition_Probability'] >= threshold]

        st.write(f"Employees with Attrition Probability ≥ {threshold}:")
        st.dataframe(filtered_data)

        # Download filtered data CSV
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_employees.csv',
            mime='text/csv',
        )

    else:
        st.info("Please upload a CSV file to get predictions.")

# Footer
st.markdown("---")
st.markdown(f"Developed by {AUTHOR} | Powered by Streamlit and scikit-learn")
