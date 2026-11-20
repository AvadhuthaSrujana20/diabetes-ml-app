from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows where target HbA1c is missing (if any)
    df = df.dropna(subset=["HbA1c"]).copy()

    # -----------------------------
    # 1. CLASSIFICATION PIPELINE
    # Target: High Diabetes Risk (HbA1c >= 6.5 -> 1, else 0)
    # -----------------------------
    y_clf = (df["HbA1c"] >= 6.5).astype(int)
    drop_for_clf = [c for c in ["HbA1c", "Fasting_Blood_Glucose"] if c in df.columns]
    X_clf = df.drop(columns=drop_for_clf)

    num_cols_clf = X_clf.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols_clf = X_clf.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf_candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
    }

    best_clf_name = ""
    best_clf_pipe = None
    best_clf_f1 = -1.0

    print("=== CLASSIFICATION RESULTS (HbA1c >= 6.5) ===")
    for name, estimator in clf_candidates.items():
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(num_cols_clf, cat_cols_clf)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train_c, y_train_c)
        preds = pipe.predict(X_test_c)
        acc = accuracy_score(y_test_c, preds)
        f1 = f1_score(y_test_c, preds, zero_division=0)
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_name = name
            best_clf_pipe = pipe

    # -----------------------------
    # 2. REGRESSION PIPELINE
    # Target: HbA1c (continuous)
    # -----------------------------
    y_reg = df["HbA1c"].astype(float)
    X_reg = df.drop(columns=["HbA1c"])

    num_cols_reg = X_reg.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols_reg = X_reg.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_candidates = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    }

    best_reg_name = ""
    best_reg_pipe = None
    best_reg_mae = float("inf")

    print("\n=== REGRESSION RESULTS (HbA1c) ===")
    for name, estimator in reg_candidates.items():
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(num_cols_reg, cat_cols_reg)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train_r, y_train_r)
        preds = pipe.predict(X_test_r)
        mae = mean_absolute_error(y_test_r, preds)
        r2 = r2_score(y_test_r, preds)
        print(f"{name}: MAE={mae:.4f}, R2={r2:.4f}")
        if mae < best_reg_mae:
            best_reg_mae = mae
            best_reg_name = name
            best_reg_pipe = pipe

    # Save best pipelines + metadata
    joblib.dump(
        {
            "pipeline": best_clf_pipe,
            "model_name": best_clf_name,
            "feature_columns": X_clf.columns.tolist(),
        },
        MODELS_DIR / "classifier_model.joblib",
    )
    joblib.dump(
        {
            "pipeline": best_reg_pipe,
            "model_name": best_reg_name,
            "feature_columns": X_reg.columns.tolist(),
        },
        MODELS_DIR / "regressor_model.joblib",
    )
    print(f"\nSaved best classifier ({best_clf_name}) & regressor ({best_reg_name}) to {MODELS_DIR}")


if __name__ == "__main__":
    main()