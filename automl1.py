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
import joblib

class SessionState:
    def __init__(self):
        self.run_button = False
        self.new_data = {}

state = SessionState()

def plot_histogram(residuals):
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    return plt

st.title("AutoML Linear Regression")
uploaded_file = st.file_uploader("Upload your clean dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]  
    cv_value = st.slider("Select the number of cross-validation folds:", min_value=2, max_value=10, value=4)
    test_size_value = st.slider("Select the test size ratio (Train-Test Split):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=42)

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
        # Add a "Run" button to trigger model fitting and prediction
        state.run_button = st.button("Run")

        if state.run_button:
            # Perform standard scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            best_model = None
            best_rmse = float('inf')
            best_model_instance = None  

            st.subheader("RMSE Scores for Selected Models:")
            for model_name, model in models:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_value, scoring='neg_mean_squared_error')
                rmse_scores = np.sqrt(-cv_scores)
                mean_rmse = np.mean(rmse_scores)

                st.write(f"{model_name}: {mean_rmse}")

                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_model = model_name
                    best_model_instance = model  # Update the best model instance

            # Display best model
            st.subheader("Best Model on Training Dataset:")
            st.write(best_model)

            # Train the best model on the full training dataset
            best_model_instance.fit(X_train, y_train)

            # Test the best model on y_test
            y_pred = best_model_instance.predict(X_test)

            # Calculate RMSE on test data
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.subheader(f"RMSE on Test Data ({best_model}):")
            st.write(test_rmse)

            # Calculate R-squared
            r_squared = r2_score(y_test, y_pred)
            st.subheader(f"R-squared on Test Data ({best_model}):")
            st.write(f"R-squared: {r_squared:.4f}")

            # Calculate residuals
            residuals = y_test - y_pred

            # Display histogram of residuals
            st.subheader("Histogram of Residuals:")
            st.pyplot(plot_histogram(residuals))

            