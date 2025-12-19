import streamlit as st
import pandas as pd
data = pd.read_csv("SelfPracticeSalary_Data.csv")
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(data[['YearsExperience']])
Y=scaler.fit_transform(data[['Salary']])
st.title("Salary Prediction App")
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('salary_prediction_model.h5', compile=False)
years_experience = st.number_input("Enter Years of Experience:",min_value=0.0, max_value=50.0, step=0.1)
if st.button("Predict Salary"):
    input_data = np.array([[years_experience]])
    predicted_salary = scaler.inverse_transform(model.predict(input_data))
    st.write(f"Predicted Salary: ${predicted_salary[0][0]:.2f}")