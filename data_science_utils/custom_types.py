"""

"""
import numpy as np
from typing import Protocol


class Model(Protocol):
    def fit(self, X, y):
        ...

    def predict(self, X) -> np.ndarray:
        ...

    def predict_proba(self, X) -> np.ndarray:
        ...

    def score(self, X, y):
        ...
