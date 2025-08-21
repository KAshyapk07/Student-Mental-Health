import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def custom_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    def _norm(x):  
        if pd.isna(x):
            return None
        return str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    def _map_col(series, mapping):

        norm_map = {k.lower(): v for k, v in mapping.items()}
        return series.map(lambda x: norm_map.get(_norm(x), pd.NA))


    yes_no_map = {"Yes": 1, "No": 0}
    for col in ["Consultation_History", "Medication_Usage"]:
        if col in df.columns:
            df[col] = _map_col(df[col], yes_no_map)


    if "Smoking_Habit" in df.columns:
        smoking_map = {
            "nonsmoker": 0,
            "occasionalsmoker": 1,
            "regularsmoker": 2,
            "heavysmoker": 3,
        }
        df["Smoking_Habit"] = _map_col(df["Smoking_Habit"], smoking_map)


    if "Alcohol_Consumption" in df.columns:
        alcohol_map = {
            "nondrinker": 0,
            "socialdrinker": 1,
            "regulardrinker": 2,
            "heavydrinker": 3,
        }
        df["Alcohol_Consumption"] = _map_col(df["Alcohol_Consumption"], alcohol_map)

    if "Diet_Quality" in df.columns:
        diet_map = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
        df["Diet_Quality"] = _map_col(df["Diet_Quality"], diet_map)

    for c in ["Consultation_History","Medication_Usage",
              "Smoking_Habit","Alcohol_Consumption","Diet_Quality"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

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
