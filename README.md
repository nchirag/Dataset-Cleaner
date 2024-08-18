# Dataset Cleaner
by Chirag N

This Streamlit app allows users to upload a dataset, clean it column by column, and download the cleaned dataset. The app provides various data cleaning options, including handling missing values, removing duplicates, managing outliers, standardizing data, and encoding categorical data.

## Features

- **Upload Dataset**: Upload CSV files to be cleaned.
- **Dataset Overview**: View initial dataset dimensions and a preview of the data.
- **Issue Detection**: Automatically detect and list issues such as missing values, duplicates, and outliers for each column.
- **Column Cleaning**: Apply various cleaning methods to each column based on the detected issues.
- **Data Cleaning Methods**:
  - Handle Missing Values: Impute with Mean, Median, Mode, or Drop Column
  - Remove Duplicates
  - Handle Outliers: Remove or Cap Outliers
  - Standardize Data
  - Encode Categorical Data: One-Hot Encoding or Label Encoding
- **Download Cleaned Dataset**: Download the cleaned dataset in CSV format.

## Installation

To run this app, you'll need Python and the following libraries. You can install the necessary libraries using `pip`:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/nchirag/Dataset-Cleaner.git
   cd Dataset-Cleaner
