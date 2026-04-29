# src/train.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from src.pipeline import build_lr_pipeline, build_rf_pipeline


def get_lr_param_grid() -> dict:
    """
    Hyperparameter grid for Logistic Regression.

    C → regularization strength
        smaller C = stronger regularization (simpler model)
        larger C  = weaker regularization (complex model)

    solver → algorithm used to optimize
        liblinear → good for small datasets
        lbfgs     → good for larger datasets
    """
    return {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['liblinear', 'lbfgs'],
        'classifier__max_iter': [1000]
    }


def get_rf_param_grid() -> dict:
    """
    Hyperparameter grid for Random Forest.

    n_estimators → number of trees
                   more trees = better but slower

    max_depth    → how deep each tree grows
                   None = unlimited (can overfit)

    min_samples_split → minimum samples to split a node
                        higher = simpler trees
    """
    return {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5]
    }


def train_with_gridsearch(pipeline, param_grid,
                          X_train, y_train,
                          model_name: str) -> GridSearchCV:
    """
    Trains a pipeline using GridSearchCV.

    scoring='f1' because dataset is imbalanced.
    Accuracy would be misleading here.

    cv=5 → 5 fold cross validation
    n_jobs=-1 → use all CPU cores (faster)
    verbose=2 → print progress
    """
    print(f"\n{'=' * 50}")
    print(f"Training {model_name} with GridSearchCV...")
    print(f"{'=' * 50}")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best Parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"\n✅ Best CV F1 Score: "
          f"{grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(grid_search, X_test,
                   y_test, model_name: str):
    """
    Evaluates the best model from GridSearchCV
    on unseen test data.
    Prints full classification report.
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluation Report — {model_name}")
    print(f"{'=' * 50}")

    y_pred = grid_search.best_estimator_.predict(X_test)

    print(classification_report(y_test, y_pred,
                                target_names=['No Churn', 'Churned']))