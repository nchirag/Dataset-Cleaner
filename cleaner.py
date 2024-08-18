import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

def detect_issues(df):
    issues = {}
    
    for column in df.columns:
        column_issues = []
        
        # Missing values
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            column_issues.append(f"Missing values: {missing_count}")
        
        # Duplicates
        duplicate_count = df.duplicated(subset=[column]).sum()
        if duplicate_count > 0:
            column_issues.append(f"Duplicate values: {duplicate_count}")
        
        # Outliers
        if pd.api.types.is_numeric_dtype(df[column]):
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
            if outlier_count > 0:
                column_issues.append(f"Outliers detected: {outlier_count}")
        
        if column_issues:
            issues[column] = column_issues
    
    return issues

def clean_column(df, column_name, column_index):
    st.subheader(f"Cleaning '{column_name}' column (Data Type: {df[column_name].dtype})")
    
    options = st.multiselect(
        "Select cleaning methods for this column:",
        get_cleaning_methods(df[column_name].dtype),
        key=f"options_{column_index}"
    )
    
    if "Handle Missing Values" in options:
        method = st.radio("Choose method for missing values:", ["Impute with Mean", "Impute with Median", "Impute with Mode", "Drop"], key=f"missing_values_{column_index}")
        if pd.api.types.is_numeric_dtype(df[column_name]):
            if method == "Impute with Mean":
                df[column_name].fillna(df[column_name].mean(), inplace=True)
            elif method == "Impute with Median":
                df[column_name].fillna(df[column_name].median(), inplace=True)
            elif method == "Impute with Mode":
                df[column_name].fillna(df[column_name].mode()[0], inplace=True)
            elif method == "Drop":
                df.dropna(subset=[column_name], inplace=True)
        else:
            st.warning(f"Cannot impute missing values for non-numeric column '{column_name}'.")
    
    if "Remove Duplicates" in options:
        df.drop_duplicates(subset=[column_name], inplace=True)
    
    if "Handle Outliers" in options and pd.api.types.is_numeric_dtype(df[column_name]):
        method = st.radio("Choose method for outliers:", ["Remove Outliers", "Cap Outliers"], key=f"outliers_{column_index}")
        if method == "Remove Outliers":
            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[column_name] >= (q1 - 1.5 * iqr)) & (df[column_name] <= (q3 + 1.5 * iqr))]
        elif method == "Cap Outliers":
            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            iqr = q3 - q1
            df[column_name] = df[column_name].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    
    if "Standardize Data" in options and pd.api.types.is_numeric_dtype(df[column_name]):
        df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    
    if "Encode Categorical Data" in options and (pd.api.types.is_categorical_dtype(df[column_name]) or df[column_name].dtype == 'object'):
        encoding_method = st.selectbox(
            "Choose encoding method:",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"encoding_{column_index}"
        )
        if encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=[column_name], drop_first=True)
        elif encoding_method == "Label Encoding":
            le = LabelEncoder()
            df[column_name] = le.fit_transform(df[column_name])
    
    return df

def get_cleaning_methods(dtype):
    methods = ["Handle Missing Values", "Remove Duplicates"]
    if pd.api.types.is_numeric_dtype(dtype):
        methods.extend(["Handle Outliers", "Standardize Data"])
    if pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
        methods.append("Encode Categorical Data")
    return methods

def load_data(uploaded_file):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    st.error("Unable to decode the file with available encodings.")
    return None

def main():
    st.title("Dataset Cleaner")
    st.write("by Chirag N")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Display dataset dimensions
            st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            
            st.write("Dataset preview:")
            st.write(df.head())
            
            # Detect issues
            issues = detect_issues(df)
            st.subheader("Detected Issues")
            if issues:
                for column, column_issues in issues.items():
                    st.write(f"**{column}**:")
                    for issue in column_issues:
                        st.write(f"- {issue}")
            else:
                st.write("No issues detected.")
            
            st.write("Select columns to clean:")
            for index, column in enumerate(df.columns):
                df = clean_column(df, column, index)
            
            st.write("Cleaned dataset preview:")
            st.write(df.head())
            
            # Download button
            towrite = BytesIO()
            df.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Download Cleaned Dataset",
                data=towrite,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
