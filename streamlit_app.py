import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the model and scaler
regmodel = pickle.load(open('linearRegressionModel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Set the title and page icon
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Define app header
st.title('Boston House Price Prediction')

# Define app sidebar
st.sidebar.header("Input Features")

# Create input fields for user to enter data
crim = st.sidebar.number_input('CRIM', min_value=0.0)
zn = st.sidebar.number_input('ZN', min_value=0.0)
indus = st.sidebar.number_input('INDUS', min_value=0.0)
chas = st.sidebar.selectbox('CHAS', [0, 1])
nox = st.sidebar.number_input('NOX', min_value=0.0)
rm = st.sidebar.number_input('RM', min_value=0.0)
age = st.sidebar.number_input('AGE', min_value=0)
dis = st.sidebar.number_input('DIS', min_value=0.0)
rad = st.sidebar.number_input('RAD', min_value=0)
tax = st.sidebar.number_input('TAX', min_value=0)
ptratio = st.sidebar.number_input('PTRATIO', min_value=0.0)
b = st.sidebar.number_input('B', min_value=0.0)
lstat = st.sidebar.number_input('LSTAT', min_value=0.0)

# Create a dictionary with the user input data
data = {
    'CRIM': crim,
    'ZN': zn,
    'INDUS': indus,
    'CHAS': chas,
    'NOX': nox,
    'RM': rm,
    'AGE': age,
    'DIS': dis,
    'RAD': rad,
    'TAX': tax,
    'PTRATIO': ptratio,
    'B': b,
    'LSTAT': lstat
}

# Predict button to make predictions
if st.sidebar.button('Predict'):
    # Transform the user input data and make predictions
    input_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = regmodel.predict(input_data)[0]
    st.success(f"The predicted house price is: ${prediction:,.2f}")

# App customization
st.sidebar.markdown("---")
st.sidebar.info("This app predicts house prices in Boston based on input features.")
st.sidebar.markdown("---")




# Run the app
if __name__ == "__main__":
    st.run()
