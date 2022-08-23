import pandas as pd
import numpy as np

import preprocessing


# Read in the values for abalone and output as a Pandas Dataframe
# Label: 'Rings' column
# Feature Notes:
# 'Sex' in this dataset is a nominal value and should be one-hot encoded
def prepare_abalone(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["Sex",
                 "Length",
                 "Diameter",
                 "Height",
                 "Whole weight",
                 "Shucked weight",
                 "Viscera weight",
                 "Shell weight",
                 "Rings"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    if display_mode:
        print("Before One-Hot Encoding:")
        print(df.head())
        print("")

    # Pre-processing steps
    df = preprocessing.encode_onehot(df, "Sex")

    if display_mode:
        print("After One-Hot Encoding:")
        print(df.head())
        print("")

    # Return the constructed Dataframe
    return df


# Read in the values for breast_cancer_wisconsin and output as a Pandas Dataframe
# Label: 'Class' column
# Feature Notes:
# 'Bare Nuclei' contains missing values denoted by '?', which this function replaces with NaN values
def prepare_breast_cancer_wisconsin(csv_file, display_mode=False):

    # Declare the column names
    col_names = ["Sample code number",
                 "Clump Thickness",
                 "Uniformity of Cell Size",
                 "Uniformity of Cell Shape",
                 "Marginal Adhesion",
                 "Single Epithelial Cell Size",
                 "Bare Nuclei",
                 "Bland Chromatin",
                 "Normal Nucleoli",
                 "Mitoses",
                 "Class"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Drop the Sample code number
    df.drop(columns=['Sample code number'], inplace=True)

    # Show missing data if in display mode
    if display_mode:
        print("Data types:")
        print(df.dtypes)
        n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'{n_missing} ? values in Bare Nuclei')
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Replace the '?' missing values with NaN
    df.replace("?", np.nan, inplace=True)

    # Per the .names file, all the columns will have integer values
    # So, convert all the columns to the numpy Int64 data type to allow for NaN values in the column
    for col in df.columns:
        df[col] = df[col].astype('Int64')

    # Show missing data if in display mode - all columns should be ints and there should be null values
    if display_mode:
        print("Data types:")
        print(df.dtypes)
        n_missing = len(df.loc[df['Bare Nuclei'] == '?'])
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'{n_missing} ? values in Bare Nuclei')
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Pre-processing steps
    preprocessing.impute_missing_vals_in_column_int64(df, 'Bare Nuclei')

    # Show missing data if in display mode - there should be no more null values
    if display_mode:
        n_null = df['Bare Nuclei'].isnull().any()
        print(f'Are null values in Bare Nuclei? {n_null}\n')

    # Return the constructed Dataframe
    return df


# Read in the values for car and output as a Pandas Dataframe
# Label: 'CAR' column
# Feature Notes:
# All of the columns in this dataset are ordinal values and should be encoded as integers as follows:
# buying: "low"->0, "med"->1, "high"->2, "vhigh"->3
# maint: "low"->0, "med"->1, "high"->2, "v-high"->3
# doors: "2"->0, "3"->1, "4"->2, "5more"->3
# persons: "2"->0, "4"->1, "more"->2
# lug_boot: "small"->0, "med"->1, "big"->2
# safety: "low"->0, "med"->1, "high"->2
def prepare_car(csv_file):

    # Declare the column names
    col_names = ["buying",
                 "maint",
                 "doors",
                 "persons",
                 "lug_boot",
                 "safety",
                 "CAR"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Pre-processing steps
    preprocessing.encode_ordinal(df, "buying", {"low": 0, "med": 1, "high": 2, "vhigh": 3})
    preprocessing.encode_ordinal(df, "maint", {"low": 0, "med": 1, "high": 2, "vhigh": 3})
    preprocessing.encode_ordinal(df, "doors", {"2": 0, "3": 1, "4": 2, "5more": 3})
    preprocessing.encode_ordinal(df, "persons", {"2": 0, "4": 1, "more": 2})
    preprocessing.encode_ordinal(df, "lug_boot", {"small": 0, "med": 1, "big": 2})
    preprocessing.encode_ordinal(df, "safety", {"low": 0, "med": 1, "high": 2})

    # Return the constructed Dataframe
    return df


# Read in the values for forestfires and output as a Pandas Dataframe
# The column headers are included in the data file and do not need to be specified here
# Label: 'area' column
# Feature Notes:
# 'month' and 'day' are ordinal values and should be encoded as integers as follow
# month: "jan"->0, "feb"->1, ... , "dec"->11
# day: "sun"->0, "mon"->1, ..., "sat"->6
def prepare_forestfires(csv_file):

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file)

    # Pre-processing steps
    preprocessing.encode_ordinal(df, "month", {"jan": 0, "feb": 1, "mar": 2, "apr": 3,
                                             "may": 4, "jun": 5, "jul": 6, "aug": 7,
                                             "sep": 8, "oct": 9, "nov": 10, "dec": 11})
    preprocessing.encode_ordinal(df, "day", {"sun": 0, "mon": 1, "tue": 2, "wed": 3,
                                           "thu": 4, "fri": 5, "sat": 6})

    # Return the constructed Dataframe
    return df


# Read in the values for house-votes-84 and output as a Pandas Dataframe
# Label: 'party' column
# Feature Notes:
# The features in this dataset are all nominal since they can be 'yea', 'nay', or '?' and should be one-hot encoded
def prepare_house_votes_84(csv_file):

    # Declare the column names
    col_names = ["party",
                 "handicapped-infants",
                 "water-project-cost-sharing",
                 "adoption-of-the-budget-resolution",
                 "physician-fee-freeze",
                 "el-salvador-aid",
                 "religious-groups-in-schools",
                 "anti-satellite-test-ban",
                 "aid-to-nicaraguan-contras",
                 "mx-missile",
                 "immigration",
                 "synfuels-corporation-cutback",
                 "education-spending",
                 "superfund-right-to-sue",
                 "crime",
                 "duty-free-exports",
                 "export-administration-act-south-africa"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Pre-processing steps
    for column in df.columns:
        if column != "party":
            df = preprocessing.encode_onehot(df, column)

    # Return the constructed Dataframe
    return df


# Read in the values for machine and output as a Pandas Dataframe
# Label: 'PRP' column
# Feature Notes:
# The vendor name column is nominal data and should be one-hot encoded
def prepare_machine(csv_file):

    # Declare the column names
    col_names = ["vendor name",
                 "Model Name",
                 "MYCT",
                 "MMIN",
                 "MMAX",
                 "CACH",
                 "CHMIN",
                 "CHMAX",
                 "PRP",
                 "ERP"]

    # Read the csv into the dataframe
    df = pd.read_csv(csv_file, names=col_names)

    # Pre-processing steps
    df = preprocessing.encode_onehot(df, 'vendor name')
    df.drop(columns=['Model Name'], inplace=True)

    # Return the constructed Dataframe
    return df