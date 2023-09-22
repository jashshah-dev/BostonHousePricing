import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot histograms and return the figure
def plot_histogram(residuals):
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    return plt

# Title for the app
st.title("Linear Regression Model Comparison")

# File upload
uploaded_file = st.file_uploader("Upload your clean dataset (CSV file):", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Data preview
    st.subheader("Data Preview")
    st.write(data.head())

    # Split data into dependent (y) and independent (X) variables
    X = data.iloc[:, :-1]  # Assuming the last column is the target variable
    y = data.iloc[:, -1]  # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Checkbox for model selection
    st.subheader("Select Linear Regression Models:")
    linear_regression = st.checkbox("Linear Regression")
    ridge = st.checkbox("Ridge")
    random_forest = st.checkbox("RandomForestRegressor")
    xgboost = st.checkbox("XGBoostRegressor")

    models = []

    if linear_regression:
        models.append(("Linear Regression", LinearRegression()))
    if ridge:
        ridge_alphas = [0.01, 0.1, 1, 10]  # You can customize the list of alphas
        for alpha in ridge_alphas:
            models.append((f"Ridge (alpha={alpha})", Ridge(alpha=alpha)))
    if random_forest:
        models.append(("RandomForestRegressor", RandomForestRegressor()))
    if xgboost:
        models.append(("XGBoostRegressor", XGBRegressor()))

    if not models:
        st.warning("Please select at least one model.")
    else:
        # Perform standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))

        best_model = None
        best_rmse = float('inf')
        best_model_instance = None  # Store the best model instance

        st.subheader("RMSE Scores for Selected Models:")
        for model_name, model in models:
            cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=4, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)

            # Calculate mean RMSE
            mean_rmse = np.mean(rmse_scores)

            st.write(f"{model_name}: {mean_rmse}")

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = model_name
                best_model_instance = model  # Update the best model instance

        # Display best model
        st.subheader("Best Model on Training Dataset:")
        st.write(best_model)

        