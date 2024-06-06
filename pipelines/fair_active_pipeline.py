"""
A pipeline is a sequence of steps starting with the downloading of data


1. Construct Data [Data 1, Data 2, ..., Data d]
2. Construct Models [Model 1, ..., Model m]
3. Evaluate Models [Metric 1, ..., Metric k]

"""

import pandas as pd
from datasets.compas import download_compas
from fair_active_learning.active_learning import active_learning
from estimators.classifiers import classifiers
from model_evaluation.classifiers import standard_scorers

active_learning_pipeline = {
    "Dataset": {"COMPAS": download_compas},
    "Estimator": classifiers,
    "Active Learning Approach": {"Active Learning": active_learning},
    "Metric": standard_scorers,
    "Budget": range(2000),
}

active_learning_metric_df = pd.DataFrame(
    data=None,
    index=pd.MultiIndex.from_product(
        iterables=active_learning_pipeline.values(),
        names=active_learning_pipeline.keys(),
    ),
    columns=["Score"],
)
