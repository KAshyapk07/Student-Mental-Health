Mental Health Prediction
📌 Project Overview

This project aims to build an AI-based system that can:
Predict whether a student is likely to have a mental health disorder (Yes/No classification).
If Yes, predict the severity level of the disorder (e.g., Mild, Moderate, Severe).
This work is part of my MS profile-building and AI/ML journey, and will later evolve into a deployable ML solution.

📊 Dataset Information

Source:OpenDataBay
 https://www.opendatabay.com/data/healthcare/7f6ef714-9d06-495d-8c1f-15683d0d0871

Size: ~47,000 rows, multiple features related to demographics, lifestyle, and health.

Target Variables:
mental_health_disorder → Binary (Yes/No).
severity → Ordinal/Multiclass (Mild, Moderate, Severe).


Project Structure
student-mental-health/
│
├── data/
│   ├── raw/                # Original dataset
│   ├── processed/          # To store cleaned datasets
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Preprocessing steps (in progress)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Preprocessing functions
│
├── README.md               # Project documentation
└── requirements.txt        # Dependencies


🔎 Work Done Till Now
✅ Exploratory Data Analysis (EDA)
Checked shape, data types, and summary statistics.
Verified missing values → found severity has ~25,000 NaN.
Checked categorical columns like gender, diet_quality, smoking_habit, alcohol_consumption.

Decided:
gender → keep Male/Female, group “Prefer not to say” separately or drop (depending on modeling).
country → not very useful → will consider dropping.
diet_quality, smoking_habit, alcohol_consumption → will be encoded as categorical features.

✅ Preprocessing Functions (Defined in src/preprocess.py)
handle_missing_values(df)
Drops/Imputes missing values (depending on feature).
Special handling for severity (since only Dataset 2 uses it).
encode_categorical(df)
Converts categorical columns into numeric (Label Encoding / One-hot Encoding).
Examples: gender, diet_quality, smoking_habit, alcohol_consumption.
scale_numeric(df)
Standardizes/normalizes numerical columns (e.g., age, hours of sleep, GPA if present).
preprocess_pipeline(df)

One combined pipeline that runs all the above functions in sequence.

Current Status:

Functions have been defined in src/preprocess.py.

Not yet applied on the dataset → preprocessing will be executed in the next notebook 