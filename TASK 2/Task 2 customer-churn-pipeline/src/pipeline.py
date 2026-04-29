# src/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.preprocessor import (
    build_preprocessor,
    get_feature_lists,
    fix_total_charges,
    encode_target
)


def build_lr_pipeline(df) -> Pipeline:
    """
    Builds complete Logistic Regression pipeline.
    Includes preprocessor + classifier in one object.

    Logistic Regression is used as baseline model -
    simple, interpretable, fast to train.
    """
    numerical_cols, categorical_cols = get_feature_lists(df)
    preprocessor = build_preprocessor(numerical_cols,
                                      categorical_cols)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,  # enough iterations to converge
            random_state=42  # reproducibility
        ))
    ])

    return pipeline


def build_rf_pipeline(df) -> Pipeline:
    """
    Builds complete Random Forest pipeline.
    Includes preprocessor + classifier in one object.

    Random Forest is used as a stronger model -
    handles non-linear relationships, robust to outliers.
    """
    numerical_cols, categorical_cols = get_feature_lists(df)
    preprocessor = build_preprocessor(numerical_cols,
                                      categorical_cols)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,  # number of trees
            random_state=42  # reproducibility
        ))
    ])

    return pipeline