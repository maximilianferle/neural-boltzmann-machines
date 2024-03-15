import numpy as np
import pandas as pd
from data import raw_data
from pathlib import Path
import json
import fancyimpute

__raw_data = Path(raw_data.__file__).parent / "clinical.project-MMRF-COMMPASS.2024-01-29.json"
__features = ["Hemoglobin",
              "Calcium",
              "Creatinine",
              "Lactate_Dehydrogenase",
              "Albumin",
              "Beta_2_Microglobulin",
              "M_Protein",
              "Serum_Free_Immunoglobulin_Light_Chain,_Lambda",
              "Serum_Free_Immunoglobulin_Light_Chain,_Kappa",
              "Leukocytes", ], ["Hb",
                                "Ca",
                                "Cr",
                                "LDH",
                                "Alb",
                                "b2m",
                                "mP",
                                "SFL-Kappa",
                                "SFL-Lambda",
                                "Lk", ]
__rename = dict(zip(*__features))


def load_raw_data():
    with open(__raw_data) as f:
        clinical_data = json.load(f)
    return clinical_data


def collect_data():
    # Assuming `data` is the loaded JSON data
    data = load_raw_data()

    # Initialize an empty list to collect the processed data
    processed_data = []

    # Loop through each case
    for case in data:
        case_id = case['case_id']
        # Loop through each follow-up
        for follow_up in case['follow_ups']:
            if 'molecular_tests' not in follow_up.keys():
                continue
            # Create a dictionary for each follow-up
            follow_up_data = {'case_id': case_id, 'days_to_follow_up': follow_up['days_to_follow_up']}
            # Add test results, ensuring spaces are replaced with underscores in the keys
            for test in follow_up['molecular_tests']:
                key = test['laboratory_test'].replace(' ', '_')
                value = test['test_value']
                unit = test['test_units']
                # Create entries for both value and unit
                follow_up_data[f'{key}'] = value
                follow_up_data[f'{key}_unit'] = unit
            # Append the dictionary to the list of processed data
            processed_data.append(follow_up_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(processed_data)
    return df


def discard_unit_columns(df):
    # Filter the columns that end with '_unit'
    unit_columns = df.filter(regex='_unit$').columns

    # Drop these columns from the DataFrame
    df_cleaned = df.drop(columns=unit_columns)
    return df_cleaned


def transform_data():
    df = collect_data()
    df = df[['case_id', 'days_to_follow_up', *__features[0]]].rename(columns=__rename)

    # Ensure that all columns that appeared in any record are present in the DataFrame
    df = df.reindex(columns=pd.unique(df.columns.ravel()), fill_value=pd.NA).sort_values(
        ['case_id', 'days_to_follow_up'], ascending=[True, True])

    df["days_to_follow_up"] = (df["days_to_follow_up"] / 30.4375).apply(round)
    df = df.rename(columns={"days_to_follow_up": "months_to_follow_up"})
    return df


def get_filled_df():
    df = transform_data()
    features = df.drop(columns=["case_id", "months_to_follow_up"]).columns.to_list()
    df = df.dropna(subset=features, how="all")
    df_filled = df.groupby('case_id').apply(lambda group: group.ffill()).apply(lambda group: group.bfill()).reset_index(
        drop=True)
    return df_filled


def get_imputed_df(imputer_method: str = "KNN",
                   imputer_params: dict = {"k": 16, },
                   ):
    # Import the respective imputation class from fancyimpute
    try:
        ImputerClass = getattr(fancyimpute, imputer_method)
    except AttributeError:
        raise ImportError(f"Could not find an imputer named {imputer_method} in fancyimpute.")

    imputer = ImputerClass(**imputer_params)

    df = transform_data()
    features = df.drop(columns=["case_id", "months_to_follow_up"]).columns.to_list()
    df = df.dropna(subset=features, how="all")

    data_array = df[features].to_numpy()
    feature_mean = np.nanmean(data_array, axis=0, keepdims=True)
    feature_std = np.nanstd(data_array, axis=0, keepdims=True)

    data_array = (data_array - feature_mean) / feature_std

    df[features] = imputer.fit_transform(data_array) * feature_std + feature_mean
    return df


def make_sequence_data(df: pd.DataFrame):
    features = df.drop(columns=["case_id", "months_to_follow_up"]).columns.to_list()
    return [df.loc[df["case_id"] == case_id, features].to_numpy() for case_id in df["case_id"].unique()]
