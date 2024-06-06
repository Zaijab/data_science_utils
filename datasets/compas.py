"""
This file will contain
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def download_compas(
    path="https://raw.githubusercontent.com/anahideh/FAL--Fair-Active-Learning/master/FAL/RecidivismData_Normalized.csv",
    attributes=[
        "MarriageStatus",
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "days_b_screening_arrest",
        "c_days_from_compas",
        "c_charge_degree",
    ],
    response_column="two_year_recid",
    sensitive_attribute=["race"],
    active_learning=False,
):
    """
    This function downloads the COMPAS dataset from Propublica.
    This is a widely used dataset in the context of fairness.
    The main controversies surrounding this dataset is the high false positive rate of models predicting "Two year recidivism" (How likely a person is to be convicted in two years).
    """
    df = pd.read_csv(path)
    df.set_index(["UID", *sensitive_attribute], inplace=True)
    X, y = df[attributes], df[response_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    if active_learning:
        labelled_index = X_train.sample(n=100).index.to_list()
        return X_train, y_train, labelled_index, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test
