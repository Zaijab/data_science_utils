"""

"""
from typing import Dict, Callable, Any
from custom_types import Model
from sklearn.metrics import get_scorer
import pandas as pd
from sklearn.linear_model import LogisticRegression

standard_scorers: Dict[str, Callable[[Model, Any, Any], float]] = {
    "Accuracy": get_scorer("accuracy"),
    "Precision": get_scorer("precision"),
    "Recall": get_scorer("recall"),
    "F1": get_scorer("f1"),
    "AUROC": get_scorer("roc_auc"),
}

standard_scorers: Dict[str, Callable[[Model, Any, Any], float]] = {
    "": get_scorer("accuracy"),
    "Precision": get_scorer("precision"),
    "Recall": get_scorer("recall"),
    "F1": get_scorer("f1"),
    "AUROC": get_scorer("roc_auc"),
}


def score_estimators(metrics, estimators, X, y):
    """"""
    metric_df = pd.DataFrame(
        data=[[]],
        index=[estimators, metrics],
    )
    return metric_df
