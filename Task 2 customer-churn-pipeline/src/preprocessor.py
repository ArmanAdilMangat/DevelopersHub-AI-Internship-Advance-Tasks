# src/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges contains whitespace strings for new customers.
    Replace with 0.0 and convert to float64.
    """
    df = df.copy()  # never modify original dataframe
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target variable Churn from Yes/No to 1/0.
    """
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


def get_feature_lists(df: pd.DataFrame):
    """
    Returns numerical and categorical feature lists.
    Excludes customerID (identifier) and Churn (target).
    TotalCharges is manually added to numerical after dtype fix.
    """
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    categorical_cols = df.select_dtypes(
        include='object').columns.tolist()

    # Remove non-features
    for col in ['customerID', 'Churn']:
        if col in categorical_cols:
            categorical_cols.remove(col)

    # TotalCharges was object - remove from categorical if present
    if 'TotalCharges' in categorical_cols:
        categorical_cols.remove('TotalCharges')

    return numerical_cols, categorical_cols


def build_numerical_transformer() -> Pipeline:
    """
    Pipeline for numerical features.
    StandardScaler handles different ranges and outliers well.
    """
    return Pipeline(steps=[
        ('scaler', StandardScaler())
    ])


def build_categorical_transformer() -> Pipeline:
    """
    Pipeline for categorical features.
    handle_unknown='ignore' prevents errors on unseen
    categories during prediction.
    """
    return Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False))
    ])


def build_preprocessor(numerical_cols: list,
                       categorical_cols: list) -> ColumnTransformer:
    """
    Combines numerical and categorical transformers
    into a single ColumnTransformer.

    remainder='drop' removes columns we don't need
    (like customerID if accidentally left in)
    """
    numerical_transformer = build_numerical_transformer()
    categorical_transformer = build_categorical_transformer()

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor