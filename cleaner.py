import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import psycopg2

# Function to connect to PostgreSQL
def connect_to_postgres():
    try:
        conn = psycopg2.connect(
            dbname="",  # Adjust database name
            user="",  # Replace with your database user
            password="",  # Replace with your database password
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

# Function to upload cleaned data to PostgreSQL
def upload_to_postgres(df, table_name="cleaned_data"):
    conn = connect_to_postgres()
    if conn:
        cursor = conn.cursor()
        try:
            # Generate column names and types
            columns = df.columns.tolist()
            column_types = [get_column_type(df[col]) for col in columns]
            
            # Create the SQL statement for creating the table
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
            create_table_query += ", ".join([f"{col} {col_type}" for col, col_type in zip(columns, column_types)])
            create_table_query += ");"
            cursor.execute(create_table_query)
            
            # Clear existing data if any
            cursor.execute(f"DELETE FROM {table_name};")

            # Insert data into the table - row by row approach
            for _, row in df.iterrows():
                values = [None if pd.isna(val) else val for val in row]
                placeholders = ', '.join(['%s' for _ in columns])
                insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders});"
                cursor.execute(insert_query, tuple(values))
            
            conn.commit()
            st.success(f"Data uploaded successfully to the '{table_name}' table!")
            return True
        except Exception as e:
            conn.rollback()
            st.error(f"Error uploading data to PostgreSQL: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    return False

# Function to get column data type for PostgreSQL
def get_column_type(column):
    if pd.api.types.is_integer_dtype(column):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(column):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(column):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(column):
        return 'TIMESTAMP'
    else:
        return 'TEXT'

# Function to detect issues in the data
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
        if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isna().all():
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

# Function to clean a column
def clean_column(df, column_name, column_index):
    st.subheader(f"Cleaning '{column_name}' column (Data Type: {df[column_name].dtype})")
    
    options = st.multiselect(
        "Select cleaning methods for this column:",
        get_cleaning_methods(df[column_name].dtype),
        key=f"options_{column_index}"
    )
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    if "Handle Missing Values" in options:
        method = st.radio("Choose method for missing values:", ["Impute with Mean", "Impute with Median", "Impute with Mode", "Drop"], key=f"missing_values_{column_index}")
        if pd.api.types.is_numeric_dtype(df_copy[column_name]):
            if method == "Impute with Mean":
                df_copy[column_name] = df_copy[column_name].fillna(df_copy[column_name].mean())
            elif method == "Impute with Median":
                df_copy[column_name] = df_copy[column_name].fillna(df_copy[column_name].median())
            elif method == "Impute with Mode":
                if not df_copy[column_name].isna().all():
                    df_copy[column_name] = df_copy[column_name].fillna(df_copy[column_name].mode()[0])
                else:
                    st.warning(f"Cannot impute with mode as all values in '{column_name}' are missing.")
            elif method == "Drop":
                df_copy = df_copy.dropna(subset=[column_name])
        else:
            if method == "Impute with Mode":
                if not df_copy[column_name].isna().all():
                    df_copy[column_name] = df_copy[column_name].fillna(df_copy[column_name].mode()[0])
                else:
                    st.warning(f"Cannot impute with mode as all values in '{column_name}' are missing.")
            elif method == "Drop":
                df_copy = df_copy.dropna(subset=[column_name])
            else:
                st.warning(f"Can only use Mode or Drop for non-numeric column '{column_name}'.")
    
    if "Remove Duplicates" in options:
        df_copy = df_copy.drop_duplicates(subset=[column_name])
    
    if "Handle Outliers" in options and pd.api.types.is_numeric_dtype(df_copy[column_name]) and not df_copy[column_name].isna().all():
        method = st.radio("Choose method for outliers:", ["Remove Outliers", "Cap Outliers"], key=f"outliers_{column_index}")
        q1 = df_copy[column_name].quantile(0.25)
        q3 = df_copy[column_name].quantile(0.75)
        iqr = q3 - q1
        if method == "Remove Outliers":
            df_copy = df_copy[(df_copy[column_name] >= (q1 - 1.5 * iqr)) | (df_copy[column_name].isna()) | 
                           (df_copy[column_name] <= (q3 + 1.5 * iqr))]
        elif method == "Cap Outliers":
            df_copy[column_name] = df_copy[column_name].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    
    if "Standardize Data" in options and pd.api.types.is_numeric_dtype(df_copy[column_name]) and not df_copy[column_name].isna().all():
        mean = df_copy[column_name].mean()
        std = df_copy[column_name].std()
        if std > 0:
            df_copy[column_name] = (df_copy[column_name] - mean) / std
        else:
            st.warning(f"Cannot standardize '{column_name}' as standard deviation is zero.")
    
    if "Encode Categorical Data" in options and (pd.api.types.is_categorical_dtype(df_copy[column_name]) or df_copy[column_name].dtype == 'object'):
        encoding_method = st.selectbox(
            "Choose encoding method:",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"encoding_{column_index}"
        )
        if encoding_method == "One-Hot Encoding":
            # Save current columns to identify new ones
            current_columns = df_copy.columns.tolist()
            df_copy = pd.get_dummies(df_copy, columns=[column_name], drop_first=True)
            # Notify user of the new column names
            new_columns = [col for col in df_copy.columns if col not in current_columns]
            if new_columns:
                st.info(f"Created new columns: {', '.join(new_columns)}")
            else:
                st.warning(f"No new columns were created after one-hot encoding '{column_name}'.")
        elif encoding_method == "Label Encoding":
            try:
                le = LabelEncoder()
                # Handle NaN values by filling with a placeholder
                has_nan = df_copy[column_name].isna().any()
                if has_nan:
                    # Create a temporary column with NaN values filled
                    temp_col = df_copy[column_name].fillna("NaN_placeholder")
                    df_copy[column_name] = le.fit_transform(temp_col)
                    # Reset NaN values after transformation
                    df_copy.loc[df_copy[column_name] == le.transform(["NaN_placeholder"])[0], column_name] = np.nan
                else:
                    df_copy[column_name] = le.fit_transform(df_copy[column_name])
            except Exception as e:
                st.error(f"Error during label encoding: {e}")
    
    return df_copy

# Function to get available cleaning methods based on data type
def get_cleaning_methods(dtype):
    methods = ["Handle Missing Values", "Remove Duplicates"]
    if pd.api.types.is_numeric_dtype(dtype):
        methods.extend(["Handle Outliers", "Standardize Data"])
    if pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
        methods.append("Encode Categorical Data")
    return methods

# Function to load data and handle encoding issues
def load_data(uploaded_file):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"File loaded successfully using {encoding} encoding.")
            return df
        except Exception as e:
            st.error(f"Error reading file with {encoding} encoding: {e}")
    return None

def download_data(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# Main Streamlit function
def main():
    st.title("Dataset Cleaner and PostgreSQL Uploader")
    st.write("By Chirag N")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            # Store raw data in session state
            st.session_state.raw_df = df
            st.session_state.cleaned_df = df.copy()

            st.write("Preview of the dataset:")
            st.write(df.head())
            
            # Show dataset description (column data types, missing values)
            st.subheader("Dataset Description:")
            st.write(df.describe())
            st.write("Missing Values Count:")
            st.write(df.isna().sum())
            
            # Detect issues in the data
            issues = detect_issues(df)
            if issues:
                st.subheader("Detected Data Issues:")
                for col, issue_list in issues.items():
                    st.write(f"**{col}:**")
                    for issue in issue_list:
                        st.write(f"- {issue}")
            
            # Cleaning data
            st.subheader("Data Cleaning:")
            for i, column_name in enumerate(df.columns):
                df = clean_column(df, column_name, i)
            
            # Save cleaned data to session state
            st.session_state.cleaned_df = df
            st.subheader("Preview of Cleaned Dataset:")
            st.write(st.session_state.cleaned_df.head())
            
            # Option to upload cleaned data to PostgreSQL
            if st.button("Upload Cleaned Data to PostgreSQL"):
                upload_to_postgres(df)
                
            # Allow user to download the cleaned dataset
            download_data(st.session_state.cleaned_df)

if __name__ == "__main__":
    main()
