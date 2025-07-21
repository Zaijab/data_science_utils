"""
"""

from custom_types import Model
from df_utils import list_loc
import pandas as pd
import numpy as np


def active_learning(
    estimator: Model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    labelled_pool: pd.MultiIndex,
    budget: int,
):
    """
    This is an implementation of active learning using Entropy.
    """
    labelled_pool = labelled_pool.copy()

    for _ in range(budget):
        unlabelled_index = X.index.difference(labelled_pool)
        unlabelled_prediction_proba = estimator.predict_proba(X.loc[unlabelled_index])
        entropy = (
            -unlabelled_prediction_proba
            * np.log2(
                unlabelled_prediction_proba, where=unlabelled_prediction_proba > 0
            )
        ).sum(axis=1)
        labelled_pool.append(
            X.loc[unlabelled_index]
            .iloc[entropy.argmax()]
            .to_frame()
            .T.index.to_list()[0]
        )
        estimator.fit(list_loc(X, labelled_pool), list_loc(y, labelled_pool))
        yield estimator, labelled_pool

    return estimator, labelled_pool
