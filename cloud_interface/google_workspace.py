"""

"""
# Set / Save Notebook - Return to working without needing to retrain the models or redownload packages
import os
import subprocess
import numpy as np
import pandas as pd
import dill


def notebook_environment(load: bool, name: str = "notebook_env.db"):

    my_bucket = os.getenv("WORKSPACE_BUCKET")

    if load:
        name_of_file_in_bucket = name
        os.system(f"gsutil cp '{my_bucket}/data/{name_of_file_in_bucket}' .")
        print(
            f"[INFO] {name_of_file_in_bucket} is successfully downloaded into your working space"
        )
        with open(name_of_file_in_bucket, "rb") as f:
            dill.load(f)
    else:
        dill.dump_session(name)
        destination_filename = name
        args = ["gsutil", "cp", f"./{destination_filename}", f"{my_bucket}/data/"]
        output = subprocess.run(args, capture_output=True)


import pickle
import os
import joblib


def save_to_workspace(pyobject, filename=None):
    """
    Given a Python Object save to Google Workspace.
    """
    with open(filename, "wb") as file:
        pickle.dump(pyobject, file)

    my_bucket = os.getenv("WORKSPACE_BUCKET")
    args = ["gsutil", "cp", f"./{filename}", f"{my_bucket}/data/"]
    output = subprocess.run(args, capture_output=True)

    return output.stderr


def load_from_workspace(filename=None, specialized=False):
    """
    Given a filename, load the data in a way that is relevant to the filename.

    match filename:

    '*.csv' => load as Pandas DataFrame
    '*.pkl' => Depickle Python Object
    '*.svg' => Display Image

    Else => Load file to "local files" and do nothing else
    """
    my_bucket = os.getenv("WORKSPACE_BUCKET")
    os.system(f"gsutil cp '{my_bucket}/data/{filename}' .")
    print(f"[INFO] {filename} is successfully downloaded into your working space")

    if specialized:
        if ".pkl" in filename:
            with open(filename, "rb") as f:
                pickle.load(f)
        elif ".csv" in filename:
            pd.read_csv(filename)
    pass


def generate_log():
    import datetime
    import time

    nb_name = "diabetes_mdpi_healthcare_validation"
    execution_time = (
        datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-10)))
        .isoformat(timespec="minutes")
        .replace("-", "_")
        .replace(":", "_")
    )
    log_file_name = f"output_{nb_name}_{execution_time}.txt"
    return log_file_name


log_file_name = generate_log()
print(f"LOGGING TO: {log_file_name}")


def log_string(string, log_file_name=log_file_name):
    import os
    import subprocess
    import datetime

    string = (
        "LOG:"
        + str(
            datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=-10))
            ).strftime("%Y-%m-%d %I:%M:%S %p")
        )
        + "\n\t"
        + string
    )
    print(string)

    destination_filename = log_file_name
    my_bucket = os.getenv("WORKSPACE_BUCKET")
    args = ["gsutil", "cp", f"./{destination_filename}", f"{my_bucket}/data/"]

    try:
        with open(log_file_name, "a") as log_file:
            log_file.writelines(str(string) + "\n")
    except exception as e:
        print("FAIL Couldn't write to log", e)

    output = subprocess.run(args, capture_output=True)


def read_log(log_file_name: str):
    import os
    import subprocess

    my_bucket = os.getenv("WORKSPACE_BUCKET")
    os.system(f"gsutil cp '{my_bucket}/data/{log_file_name}' .")
    with open(log_file_name, "r") as log_file:
        contents = log_file.read()
        return contents


def generate_log():
    import datetime
    import time

    nb_name = "diabetes_mdpi_healthcare_validation"
    execution_time = (
        datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-10)))
        .isoformat(timespec="minutes")
        .replace("-", "_")
        .replace(":", "_")
    )
    log_file_name = f"output_{nb_name}_{execution_time}.txt"
    return log_file_name


log_file_name = generate_log()
print(f"LOGGING TO: {log_file_name}")


def log_string(string, log_file_name=log_file_name):
    import os
    import subprocess
    import datetime

    string = (
        "LOG:"
        + str(
            datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=-10))
            ).strftime("%Y-%m-%d %I:%M:%S %p")
        )
        + "\n\t"
        + string
    )
    print(string)

    destination_filename = log_file_name
    my_bucket = os.getenv("WORKSPACE_BUCKET")
    args = ["gsutil", "cp", f"./{destination_filename}", f"{my_bucket}/data/"]

    try:
        with open(log_file_name, "a") as log_file:
            log_file.writelines(str(string) + "\n")
    except exception as e:
        print("FAIL Couldn't write to log", e)

    output = subprocess.run(args, capture_output=True)


def read_log(log_file_name: str):
    import os
    import subprocess

    my_bucket = os.getenv("WORKSPACE_BUCKET")
    os.system(f"gsutil cp '{my_bucket}/data/{log_file_name}' .")
    with open(log_file_name, "r") as log_file:
        contents = log_file.read()
        return contents
