import numpy as np
import pandas as pd  
import os

def fetch_mimic(data_path, return_benchmark_data=False):
    """
    Fetch the MIMIC-III dataset from the raw MIMIC-III csv files.

    Data is prepared using the following steps:

    1. For each patient, we define the index time as 48
    hours after their first admission and construct features exclusively
    from data recorded before this index time.
    
    2. Continuous and categorical features are collected from labevents and chartevents
    using all available data recorded before each patient's index
    time.
    
    3. Continuous features are averaged over the pre-index window,
    while categorical features are assigned the most recent value before
    the index time.
    
    4. Features are deemed eligible if they are observed in at least 5%
    of patients.

    The resulting dataset contains 880 features.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the MIMIC-III csv files.
        
    return_benchmark_data : bool, default=False
        If True, the function will return data with one feature for each feature in the 
        MIMIC-III benchmark data, see https://arxiv.org/pdf/1703.07771, along with the full dataset.
        If False, the function will return only, the full dataset.
        
    Returns
    -------
    pandas.DataFrame or tuple
        The cohort dataset if return_benchmark_data is False, 
        else (cohort dataset, benchmark dataset).
    """
    
    # ----------------- Step 1 -----------------
    # Set the index date
    # Set to be 48 hours after the first admisssion time.

    patients = pd.read_csv(os.path.join(data_path, "PATIENTS.csv"))
    admissions = pd.read_csv(os.path.join(data_path, "ADMISSIONS.csv"))

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])

    first_admissions = admissions.loc[
        admissions.groupby("SUBJECT_ID")["ADMITTIME"].idxmin(),
        ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ETHNICITY", "HOSPITAL_EXPIRE_FLAG"]
    ]
    cohort_first_admissions = pd.merge(patients, first_admissions, on='SUBJECT_ID', how='inner')
    cohort_first_admissions["DOB"] = pd.to_datetime(cohort_first_admissions["DOB"])
    cohort_first_admissions["ADMITTIME"] = pd.to_datetime(cohort_first_admissions["ADMITTIME"])
    cohort_first_admissions["index_time"] = cohort_first_admissions["ADMITTIME"] + pd.Timedelta("48 hours")

    # ----------------- Step 2 -----------------
    # Get features from lab events

    labevents = pd.read_csv(os.path.join(data_path, "LABEVENTS.csv"))
    labevents["CHARTTIME"] = pd.to_datetime(labevents["CHARTTIME"])
    labevents["ITEMID"] = labevents["ITEMID"].astype(int)

    # First numerical features
    labevents_with_hadm_id = labevents[labevents["HADM_ID"].notna()].copy()
    labevents_with_hadm_id["ITEMID"] = labevents_with_hadm_id["ITEMID"].astype(int)

    # Only get lab events from the first admission that are before the index time
    labevents_with_hadm_id = pd.merge(
        labevents_with_hadm_id,
        cohort_first_admissions[["SUBJECT_ID", "HADM_ID", "index_time"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner"
    )
    labevents_before_index = labevents_with_hadm_id[
        (labevents_with_hadm_id["CHARTTIME"] < labevents_with_hadm_id["index_time"])
    ]

    # Get the mean of the lab events for each patient
    labevents_numeric_before_index = labevents_before_index \
                                    .groupby(["SUBJECT_ID", "ITEMID", "VALUEUOM"])["VALUENUM"] \
                                    .mean() \
                                    .reset_index()
    labevents_numeric_before_index["ITEMID_with_UOM"] = labevents_numeric_before_index["ITEMID"].astype(str) + \
                                                        "_" + \
                                                        labevents_numeric_before_index["VALUEUOM"]
    labevents_numeric_before_index = labevents_numeric_before_index.pivot(
        index="SUBJECT_ID",
        columns="ITEMID_with_UOM",
        values="VALUENUM"
    )
    labevents_numeric_before_index.columns = ["lab_" + str(col) for col in labevents_numeric_before_index.columns]
    labevents_numeric_before_index = labevents_numeric_before_index.reset_index()
    labevents_numeric_before_index.columns.name = None

    # Remove columns with more than 95% missing values
    labevents_numeric_before_index = labevents_numeric_before_index[
        labevents_numeric_before_index.columns[
            labevents_numeric_before_index.isna().mean() < 0.95
        ]
    ]

    # Set the columns to be of type float
    for c in labevents_numeric_before_index.columns[1:]:
        labevents_numeric_before_index[c] = labevents_numeric_before_index[c].astype(float)
        
    # Now get categorical features from lab events
    labevents_categorical_before_index = labevents_before_index[labevents_before_index["VALUENUM"].isna()]
    labevents_categorical_before_index = (
        labevents_categorical_before_index.loc[
            labevents_categorical_before_index.groupby(["SUBJECT_ID", "ITEMID", "VALUEUOM"])["CHARTTIME"].idxmax(),
            ["SUBJECT_ID", "ITEMID", "VALUEUOM", "VALUE"]
        ]
        .reset_index(drop=True)
    )
    labevents_categorical_before_index["ITEMID_with_UOM"] = labevents_categorical_before_index["ITEMID"].astype(str) + \
                                                        "_" + \
                                                        labevents_categorical_before_index["VALUEUOM"]
    labevents_categorical_before_index = labevents_categorical_before_index.pivot(
        index="SUBJECT_ID",
        columns="ITEMID_with_UOM",
        values="VALUE"
    )
    labevents_categorical_before_index.columns = ["lab_" + str(col) + "_cat" for col in labevents_categorical_before_index.columns]
    labevents_categorical_before_index = labevents_categorical_before_index.reset_index()
    labevents_categorical_before_index.columns.name = None
    labevents_categorical_before_index = labevents_categorical_before_index[
        labevents_categorical_before_index.columns[
            labevents_categorical_before_index.isna().mean() < 0.95
        ]
    ]

    # ----------------- Step 3 -----------------
    # Get features from chart events

    chartevents = pd.read_csv(os.path.join(data_path, "CHARTEVENTS.csv"), low_memory=False)
    chartevents["CHARTTIME"] = pd.to_datetime(chartevents["CHARTTIME"])

    # First numerical
    # Basically the same as lab events but without units

    chartevents_with_hadm_id = chartevents[chartevents["HADM_ID"].notna()]
    chartevents_with_hadm_id["ITEMID"] = chartevents_with_hadm_id["ITEMID"].astype(int)

    chartevents_with_hadm_id = pd.merge(
        chartevents_with_hadm_id,
        cohort_first_admissions[["SUBJECT_ID", "HADM_ID", "index_time"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner"
    )
    chartevents_before_index = chartevents_with_hadm_id[
        (chartevents_with_hadm_id["CHARTTIME"] < chartevents_with_hadm_id["index_time"])
    ]

    chartevents_numeric_before_index = chartevents_before_index \
                                    .groupby(["SUBJECT_ID", "ITEMID"])["VALUENUM"] \
                                    .mean() \
                                    .reset_index()
    chartevents_numeric_before_index = chartevents_numeric_before_index.pivot(
        index="SUBJECT_ID",
        columns="ITEMID",
        values="VALUENUM"
    )
    chartevents_numeric_before_index.columns = ["chart_" + str(col) for col in chartevents_numeric_before_index.columns]
    chartevents_numeric_before_index = chartevents_numeric_before_index.reset_index()
    chartevents_numeric_before_index.columns.name = None
    chartevents_numeric_before_index = chartevents_numeric_before_index[
        chartevents_numeric_before_index.columns[
            chartevents_numeric_before_index.isna().mean() < 0.95
        ]
    ]
    for c in chartevents_numeric_before_index.columns[1:]:
        chartevents_numeric_before_index[c] = chartevents_numeric_before_index[c].astype(float)
        
    # Then categorical
    chartevents_categorical_before_index = chartevents_before_index[chartevents_before_index["VALUENUM"].isna()]

    chartevents_categorical_before_index = (
        chartevents_categorical_before_index.loc[
            chartevents_categorical_before_index.groupby(["SUBJECT_ID", "ITEMID"])["CHARTTIME"].idxmax(),
            ["SUBJECT_ID", "ITEMID", "VALUE"]
        ]
        .reset_index(drop=True)
    )
    chartevents_categorical_before_index = chartevents_categorical_before_index.pivot(
        index="SUBJECT_ID",
        columns="ITEMID",
        values="VALUE"
    )
    chartevents_categorical_before_index.columns = ["chart_" + str(col) + "_cat" for col in chartevents_categorical_before_index.columns]
    chartevents_categorical_before_index = chartevents_categorical_before_index.reset_index()
    chartevents_categorical_before_index.columns.name = None
    chartevents_categorical_before_index = chartevents_categorical_before_index[
        chartevents_categorical_before_index.columns[
            chartevents_categorical_before_index.isna().mean() < 0.95
        ]
    ]

    # ----------------- Step 4 -----------------
    # Putting it all together

    cohort_data = cohort_first_admissions[[
        "SUBJECT_ID", 
        "index_time", 
        "DOB", 
        "DOD", 
        "ADMITTIME", 
        "GENDER",
        "ETHNICITY",
        "HOSPITAL_EXPIRE_FLAG"
    ]].copy()

    cohort_data["DEATH_TIME"] = (pd.to_datetime(cohort_data["DOD"]) - pd.to_datetime(cohort_data["ADMITTIME"])).dt.total_seconds() / (3600*24)
    cohort_data["event"] = cohort_data["DEATH_TIME"].notna()

    # Get the survival time
    last_admissions = admissions.loc[
        admissions.groupby("SUBJECT_ID")["ADMITTIME"].idxmax(),
        ["SUBJECT_ID", "DISCHTIME"]
    ].rename(columns={"DISCHTIME": "LAST_DISCHTIME"})
    cohort_data = pd.merge(cohort_data, last_admissions, on="SUBJECT_ID", how='inner')
    cohort_data["CENSOR_TIME"] = (
        pd.to_datetime(cohort_data["LAST_DISCHTIME"]) - \
        pd.to_datetime(cohort_data["index_time"])
    ).dt.total_seconds() / (3600*24)
    cohort_data["time"] = cohort_data["DEATH_TIME"].fillna(cohort_data["CENSOR_TIME"])

    # Get the age
    # If the year of birth is before 1905, set the age to 89
    cohort_data["AGE"] = 89.0
    cohort_data.loc[cohort_data["DOB"].dt.year >= 1905, "AGE"] = (
        (cohort_data.loc[cohort_data["DOB"].dt.year >= 1905, "ADMITTIME"] - \
            cohort_data.loc[cohort_data["DOB"].dt.year >= 1905, "DOB"])
        .dt.days // 365.25
    )

    cohort_data = pd.merge(cohort_data, labevents_numeric_before_index, on="SUBJECT_ID", how='left')
    cohort_data = pd.merge(cohort_data, labevents_categorical_before_index, on="SUBJECT_ID", how='left')
    cohort_data = pd.merge(cohort_data, chartevents_numeric_before_index, on="SUBJECT_ID", how='left')
    cohort_data = pd.merge(cohort_data, chartevents_categorical_before_index, on="SUBJECT_ID", how='left')

    # Dropping one weird column which has times
    # That get messed up
    # cohort_data = cohort_data.drop(["chart_4171"], axis=1)


    # Remove patients with times <= 0
    cohort_data = cohort_data[cohort_data["time"] > 0]

    # Removing patients that are < 18 years old
    cohort_data = cohort_data[cohort_data["AGE"] >= 18]

    # Remove columns that are missing in > 95% of the patients
    # after concatenating the data
    cohort_data = cohort_data[
        cohort_data.columns[
            cohort_data.isna().mean() < 0.95
        ]
    ]
    
    # ----------------- Step 5 -----------------
    # Fix the column names

    items = pd.read_csv(os.path.join(data_path, "D_ITEMS.csv"))
    lab_items = pd.read_csv(os.path.join(data_path, "D_LABITEMS.csv"))

    cohort_data_with_full_col_names = cohort_data.copy()

    new_cols = []
    for col in cohort_data_with_full_col_names.columns:
        if col.startswith("lab_"):
            itemid = int(col.split("_")[1])
            unit = col.split("_")[2]
            item_name = lab_items[lab_items["ITEMID"] == itemid]["LABEL"].values[0].replace(" ", "_")
            new_cols.append(f"lab_{item_name}_{unit}")
        elif col.startswith("chart_"):
            itemid = int(col.split("_")[1])
            item_name = items[items["ITEMID"] == itemid]["LABEL"].values[0].replace(" ", "_")
            new_cols.append(f"chart_{item_name}")
        else:
            new_cols.append(col)
            
    cohort_data_with_full_col_names.columns = new_cols
    
    # Collapse any duplicate columns after renaming
    for c in cohort_data_with_full_col_names.columns:
        if len(cohort_data_with_full_col_names[[c]].columns) > 1:
            if "_cat" in c:
                # Find the most frequent
                new_col = cohort_data_with_full_col_names[[c]].mode(axis=1).iloc[:, 0]
            else:
                # Check if all columns are numeric
                if cohort_data_with_full_col_names[[c]].select_dtypes(include=[np.number]).shape[1] == 0:
                    new_col = cohort_data_with_full_col_names[[c]].mode(axis=1).iloc[:, 0]
                else:
                    new_col = cohort_data_with_full_col_names[[c]].select_dtypes(include=[np.number]).mean(axis=1)
                
            cohort_data_with_full_col_names = cohort_data_with_full_col_names.drop([c], axis=1)
            cohort_data_with_full_col_names[c] = new_col

    # Return the final cohort data
    if not return_benchmark_data:
        return cohort_data_with_full_col_names, cohort_data

    # ----------------- Step 6 -----------------
    # Make the benchmark data

    # Read in benchmark items from mimic3-benchmarks
    benchmark_items = pd.read_csv("https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/ea0314c7cbd369f62e2237ace6f683740f867e3a/mimic3benchmark/resources/itemid_to_variable_map.csv")

    benchmark_data = cohort_data[[
        'SUBJECT_ID', 'GENDER', 'ETHNICITY', 'HOSPITAL_EXPIRE_FLAG', 'AGE',
    ]].copy()

    benchmark_data["Capillary_refill_rate"] = cohort_data["chart_223951_cat"]
    benchmark_data["Capillary_refill_rate"] = np.where(
        benchmark_data["Capillary_refill_rate"].isna(),
        cohort_data["chart_224308_cat"],
        benchmark_data["Capillary_refill_rate"]
    )
    benchmark_data["Capillary_refill_rate"] = benchmark_data["Capillary_refill_rate"].replace({
        "Normal <3 Seconds": "Brisk",
        "Abnormal >3 Seconds": "Delayed",
        "Comment": np.nan
    })

    # Diastolic Blood Pressure
    benchmark_data["Diastolic_BP"] = cohort_data["chart_8368"]
    benchmark_data["Diastolic_BP"] = np.where(
        benchmark_data["Diastolic_BP"].isna(),
        cohort_data["chart_220051"],
        benchmark_data["Diastolic_BP"]
    )
    benchmark_data["Diastolic_BP"] = np.where(
        benchmark_data["Diastolic_BP"].isna(),
        cohort_data["chart_8441"],
        benchmark_data["Diastolic_BP"]
    )
    benchmark_data["Diastolic_BP"] = np.where(
        benchmark_data["Diastolic_BP"].isna(),
        cohort_data["chart_220180"],
        benchmark_data["Diastolic_BP"]
    )
    benchmark_data["Inspired_oxygen"] = cohort_data["chart_223835"]

    # GCS
    benchmark_data["GCS_Total"] = cohort_data["chart_198"]
    benchmark_data["GCS_Eye_Opening"] = np.where(
        cohort_data["chart_220739"].isna(),
        cohort_data["chart_184"],
        cohort_data["chart_220739"]
    )
    benchmark_data["GCS_Motor_Response"] = np.where(
        cohort_data["chart_223900"].isna(),
        cohort_data["chart_454"],
        cohort_data["chart_223900"]
    )
    benchmark_data["GCS_Verbal_Response"] = np.where(
        cohort_data["chart_223901"].isna(),
        cohort_data["chart_723"],
        cohort_data["chart_223901"]
    )

    # Glucose
    glucose_items = benchmark_items[benchmark_items["LEVEL2"].str.contains("Glucose", na=False)]["ITEMID"].values
    glucose_cols = []
    for c in glucose_items:
        if f"chart_{c}" in cohort_data.columns:
            glucose_cols.append(f"chart_{c}")
    benchmark_data["Glucose"] = cohort_data[glucose_cols].mean(axis=1)

    # Heart Rate
    benchmark_data["Heart_Rate"] = np.where(
        cohort_data["chart_220045"].isna(),
        cohort_data["chart_211"],
        cohort_data["chart_220045"]
    )

    # Height
    benchmark_data["Height"] = cohort_data["chart_226707"]

    # Mean Blood Pressure
    mbp_items = benchmark_items[benchmark_items["LEVEL2"].str.contains("Mean blood pressure", na=False)]["ITEMID"].values
    mbp_cols = []
    for c in mbp_items:
        if f"chart_{c}" in cohort_data.columns:
            mbp_cols.append(f"chart_{c}")
    benchmark_data["Mean_Blood_Pressure"] = cohort_data[mbp_cols].mean(axis=1)

    # Oxygen Saturation
    oxy_items = benchmark_items[benchmark_items["LEVEL2"].str.contains("Oxygen saturation", na=False)]["ITEMID"].values
    oxy_cols = []
    for c in oxy_items:
        if f"chart_{c}" in cohort_data.columns:
            oxy_cols.append(f"chart_{c}")
    benchmark_data["Oxygen_Saturation"] = cohort_data[oxy_cols].mean(axis=1)

    # Respiratory Rate
    resp_items = benchmark_items[benchmark_items["LEVEL2"].str.contains("Respiratory rate", na=False)]["ITEMID"].values
    resp_cols = []
    for c in resp_items:
        if f"chart_{c}" in cohort_data.columns:
            resp_cols.append(f"chart_{c}")
    benchmark_data["Respiratory_Rate"] = cohort_data[resp_cols].mean(axis=1)

    # Systolic Blood Pressure
    sbp_items = benchmark_items[benchmark_items["LEVEL2"].str.contains("Systolic blood pressure", na=False)]["ITEMID"].values
    sbp_cols = []
    for c in sbp_items:
        if f"chart_{c}" in cohort_data.columns:
            sbp_cols.append(f"chart_{c}")
    benchmark_data["Systolic_Blood_Pressure"] = cohort_data[sbp_cols].mean(axis=1)

    # Temperature
    temp_F_items = benchmark_items[
        benchmark_items["LEVEL1"].str.contains("Temperature", na=False) & benchmark_items["LEVEL1"].str.contains("F", na=False)
    ]["ITEMID"].values
    temp_F_cols = []
    for c in temp_F_items:
        if f"chart_{c}" in cohort_data.columns:
            temp_F_cols.append(f"chart_{c}")
    temp_C_items = benchmark_items[
        benchmark_items["LEVEL1"].str.contains("Temperature", na=False) & benchmark_items["LEVEL1"].str.contains("C", na=False)
    ]["ITEMID"].values
    temp_C_cols = []
    for c in temp_C_items:
        if f"chart_{c}" in cohort_data.columns:
            temp_C_cols.append(f"chart_{c}")
    benchmark_data["Temperature"] = cohort_data[temp_F_cols].mean(axis=1)
    benchmark_data["Temperature"] = np.where(
        benchmark_data["Temperature"].isna(),
        cohort_data[temp_C_cols].mean(axis=1) * 9/5 + 32,
        benchmark_data["Temperature"]
    )

    # Weight
    benchmark_data["Weight"] = cohort_data["chart_763"]

    # pH
    benchmark_data["pH"] = cohort_data["lab_50820_units"]

    # Write to csv
    return cohort_data_with_full_col_names, benchmark_data