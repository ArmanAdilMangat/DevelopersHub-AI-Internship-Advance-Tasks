# src/predict.py
import joblib
import pandas as pd
from src.config import PIPELINE_PATH
from src.preprocessor import fix_total_charges


def save_pipeline(pipeline, path=PIPELINE_PATH) -> None:
    """
    Exports trained pipeline to disk using joblib.
    Creates models/ directory if it doesn't exist.
    """
    path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Pipeline saved → {path}")


def load_pipeline(path=PIPELINE_PATH):
    """
    Loads exported pipeline from disk.
    Raises FileNotFoundError if pipeline doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No pipeline found at {path}. "
            f"Train and export the model first."
        )
    pipeline = joblib.load(path)
    print(f"✅ Pipeline loaded from → {path}")
    return pipeline


def predict_churn(customer_data: pd.DataFrame,
                  pipeline) -> pd.DataFrame:
    """
    Makes churn predictions on new customer data.
    Returns dataframe with prediction and probability.

    Args:
        customer_data: raw customer features (not preprocessed)
        pipeline: loaded sklearn pipeline

    Returns:
        DataFrame with Churn_Prediction and Churn_Probability
    """
    # Fix TotalCharges dtype before prediction
    customer_data = fix_total_charges(customer_data)

    # Get predictions and probabilities
    predictions = pipeline.predict(customer_data)
    probabilities = pipeline.predict_proba(
                        customer_data)[:, 1]

    # Build results dataframe
    results = customer_data.copy()
    results['Churn_Prediction'] = predictions
    results['Churn_Probability'] = probabilities.round(3)
    results['Churn_Label'] = results[
        'Churn_Prediction'].map({1: 'Will Churn ⚠️',
                                  0: 'Will Stay ✅'})

    return results[['Churn_Prediction',
                     'Churn_Probability',
                     'Churn_Label']]