from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

def perturb_datetime(input_datetime, interval, time_unit="DAY"):
    """
    Perturbs a given datetime within a specified interval.

    Args:
        input_datetime: The original datetime.
        interval (tuple): A tuple containing the minimum and maximum perturbation values.
        time_unit (str): Unit of time for perturbation, e.g., "DAY", "MONTH", "YEAR" (default: "DAY").

    Returns:
        datetime: The perturbed datetime.

    """

    # Convert interval to timedelta
    min_interval, max_interval = interval
    time_delta_multiplier = {
        "YEAR": 365,
        "MONTH": 30,
        "WEEK": 7,
        "DAY": 1
    }.get(time_unit, 1)

    # Generate a random timedelta within the interval
    random_timedelta_days = random.randint(min_interval, max_interval) * time_delta_multiplier
    random_timedelta = timedelta(days=random_timedelta_days)

    # Calculate the new datetime
    new_datetime = input_datetime + random_timedelta

    return new_datetime


def convert_one_row_to_json_dict(input_values: tuple) -> List[Dict[str, str]]:
    """
    Converts a single row of input values to a list of JSON records.

    Args:
        input_values (tuple): A tuple containing the row values, ALL_COLS, and date_cache.

    Returns:
        List[Dict[str, str]]: List of JSON records, each containing date and code.

    """

    row, ALL_COLS, date_cache = input_values
    row_dicts = []

    for col_index, value in enumerate(row):
        if value:
            column = ALL_COLS[col_index]

            # Extract age and code information from the column name
            age = "_".join(column.split("_")[1:])
            code = column.split('_')[0]

            # Perturb the date using a function perturb_datetime (not provided)
            date = perturb_datetime(
                date_cache[age],
                (-3, 3),
                "MONTHS"
            ).strftime('%Y-%m-%d')

            if value == 'X':
                row_dicts.append({"date": date, "code": code})
            else:
                # Append code with the first character of the value
                row_dicts.append({"date": date, "code": f'{code}.{value[0]}'})

    return row_dicts


def convert_teomim_to_json(df: pd.DataFrame,
                             latest_date: str = '2024-01-01',
                             id_label: str = "000",
                             id_prefix: str = "QP",
                             num_workers: int = None,
                             max_age: int = 75.5,
                             VERBOSE: bool = False) -> List[Dict[str, str]]:
    """
    Converts teomim DataFrame to a list of JSON records.

    Args:
        df (pd.DataFrame): Input DataFrame containing patient data.
        latest_date (str): Latest date to consider for age calculations (default: '2024-01-01').
        id_label (str): Label for patient IDs (default: "000").
        id_prefix (str): Prefix for patient IDs (default: "QP").
        num_workers (int): Number of parallel workers for processing rows (default: None).
        max_age (int): Maximum age to be found in the patient records. TODO REMOVE THIS AND INFER INSTEAD (default: 75.5).

        VERBOSE (bool): Flag to control verbosity (default: False).

    Returns:
        List[Dict[str, str]]: List of JSON records, each containing patient_id, birth_date, and DX_record.

    Raises:
        ValueError: If the input DataFrame is empty.

    Note:
        This function assumes that the DataFrame has specific column naming conventions related to patient age.

    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Convert latest_date to datetime object
    latest_date = datetime.strptime(latest_date, '%Y-%m-%d')

    ALL_COLS = list(df.columns)
    ALL_AGES = list(set(["_".join(i.split("_")[1:]) for i in ALL_COLS]))

    date_cache = {}
    age_cache = {}

    # Save dates of a certain age given max_age and latest_date
    for column in ALL_AGES:
        age, extension = column.split('_')
        age, extension = int(age), int(extension)
        total_months = (max_age - age) * 12 + (extension - 1) * 6
        computed_date = latest_date - timedelta(days=total_months * 30)
        date_cache[column] = computed_date
        age_cache[column] = computed_date

    # Calculate a fixed birth_date for all patients based on the maximum age
    birth_date = latest_date - timedelta(days=int(max_age * 365.25))

    # Convert DataFrame to numpy array
    np_array = df.fillna("").values.astype(str)

    total_rows = len(df)
    tqdm_bar = tqdm(total=total_rows, desc="Processing Rows", disable=not VERBOSE)

    def update_tqdm(_):
        # Update tqdm progress bar after each row is processed
        tqdm_bar.update(1)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                convert_one_row_to_json_dict,
                (row, ALL_COLS, date_cache)
            ) for row in np_array
        ]

        for future in futures:
            future.add_done_callback(update_tqdm)

        # Collect results from parallel processing
        DX_records = [future.result() for future in futures]

    tqdm_bar.close()

    json_data = []
    for i, DX_record in tqdm(
            enumerate(DX_records),
            total=len(DX_records),
            disable=not VERBOSE):
        json_data.append(
            {
                "patient_id": f"{id_prefix}{id_label}_{str(i).rjust(8, '0')}",
                "birth_date": birth_date.strftime('%Y-%m-%d'),
                'DX_record': DX_record
            }
        )

    return json_data


def convert_json_to_teomim(patient_records, columns, VERBOSE=False):
    """
    Fills a teomim dataframe with values based on patient records and provided QNet columns.

    Args:
        patient_records (list): List of dictionaries representing patient records.
        columns (list): List of column names representing CODE_AGE_EXTENSION.
        VERBOSE (bool): Flag to control verbosity for tqdm (default: False).

    Returns:
        list: 2D array filled with values based on patient records and columns.

    """
    # Initialize an empty 2D array with default values as empty strings
    patient_data_array = [["" for _ in columns] for _ in range(len(patient_records))]

    # Find the minimum and maximum ages from patient's DX_records' dates
    min_age = float('inf')
    max_age = float('-inf')

    for patient_record in patient_records:
        birth_date = datetime.strptime(patient_record['birth_date'], '%Y-%m-%d')

        for dx_record in patient_record['DX_record']:
            date = datetime.strptime(dx_record['date'], '%Y-%m-%d')
            age_at_code_recorded = (date - birth_date).days / 365.25  # Age in years at the time of code recorded
            min_age = min(min_age, age_at_code_recorded)
            max_age = max(max_age, age_at_code_recorded)

    # Iterate over patients with tqdm decorator
    with tqdm(total=len(patient_records), desc="JSON ==> QSAMPLE", disable=not VERBOSE) as pbar:
        for i, patient_record in enumerate(patient_records):
            birth_date = datetime.strptime(patient_record['birth_date'], '%Y-%m-%d')

            for dx_record in patient_record['DX_record']:
                code = dx_record['code']
                date = datetime.strptime(dx_record['date'], '%Y-%m-%d')
                age_at_code_recorded = (date - birth_date).days / 365.25  # Age in years at the time of code recorded

                # Extract CODE, AGE, and EXTENSION from the code and determine the corresponding column
                code_part = code[:3]
                age_part = int(age_at_code_recorded)
                extension = 1 if age_at_code_recorded % 1 < 0.5 else 2
                column_index = columns.index(f"{code_part}_{age_part}_{extension}") if f"{code_part}_{age_part}_{extension}" in columns else None

                # Skip if the CODE is not found among CODE parts of the columns
                if column_index is not None:
                    # Determine the value to fill in the array
                    if len(code) == 3 or '.' not in code:
                        value = "X"
                    else:
                        value = code.split('.')[1][0]

                    # Fill the array at the determined location
                    patient_data_array[i][column_index] = value

            # Fill one's column values where AGE_EXTENSION combination is between min_age and max_age, and where no code is found
            for j, column in enumerate(columns):
                _, age, extension = column.split('_')
                age, extension = int(age), int(extension)

                # Check if extension is either 1 or 2 (not just 1)
                if extension in [1, 2] and min_age <= age <= max_age and patient_data_array[i][j] == "":
                    patient_data_array[i][j] = "."

            # Update tqdm progress bar
            pbar.set_postfix(Min_Age=min_age, Max_Age=max_age)
            pbar.update(1)

    return pd.DataFrame(patient_data_array, columns = columns)
