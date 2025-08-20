import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def custom_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    binary_cols = ["Consultation_History", "Smoking_Habit",
                   "Alcohol_Consumption", "Medication_Usage"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if "Occupation" in df.columns:
        top_occ = df["Occupation"].value_counts().nlargest(10).index
        df["Occupation"] = df["Occupation"].apply(lambda x: x if x in top_occ else "Other")

    return df


def get_preprocessor(df: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    if "Gender" in df.columns:
        categorical_cols.append("Gender")
    if "Diet_Quality" in df.columns:
        categorical_cols.append("Diet_Quality")
    if "Occupation" in df.columns:
        categorical_cols.append("Occupation")

    numeric_cols = ["Age", "Stress_Level", "Sleep_Hours", "Work_Hours",
                    "Physical_Activity_Hours", "Social_Media_Usage"]

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, categorical_cols),
            ("numerical", num_pipeline, numeric_cols)
        ],
        remainder="passthrough"
    )

    return preprocessor


def create_stage_datasets(df: pd.DataFrame):
    stage1 = df.drop(columns=["Severity"])
    stage1_target = stage1.pop("Mental_Health_Condition").map({"Yes": 1, "No": 0})
    stage1["target"] = stage1_target

    stage2 = df.dropna(subset=["Severity"]).copy()
    stage2_target = stage2.pop("Severity").map({"Mild": 0, "Moderate": 1, "Severe": 2})
    stage2["target"] = stage2_target

    return stage1, stage2


def save_datasets(stage1, stage2, path_stage1, path_stage2):
    stage1.to_csv(path_stage1, index=False)
    stage2.to_csv(path_stage2, index=False)
