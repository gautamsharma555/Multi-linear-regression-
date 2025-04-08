# mlr_lasso_app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title
st.title("Toyota Corolla Price Prediction using Lasso Regression")

# Upload CSV
uploaded_file = st.file_uploader("Upload ToyotaCorolla - MLR.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Drop unnamed or irrelevant columns
    df.drop(['Id', 'Model'], axis=1, inplace=True, errors='ignore')

    # Encoding categorical variables (if any)
    df = pd.get_dummies(df, drop_first=True)

    # Handle outliers
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))

    # Features & Target
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = lasso.predict(X_train_scaled)
    y_test_pred = lasso.predict(X_test_scaled)

    # Evaluate
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    st.subheader("Model Performance")
    st.write(f"**R² (Train):** {r2_train:.4f}")
    st.write(f"**R² (Test):** {r2_test:.4f}")
    st.write(f"**MSE (Train):** {mse_train:.2f}")
    st.write(f"**MSE (Test):** {mse_test:.2f}")

    # Show top features
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lasso.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    st.subheader("Feature Importance (Lasso Coefficients)")
    st.dataframe(coef_df)
