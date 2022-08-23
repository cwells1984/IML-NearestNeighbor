import copy

import numpy as np
import pandas as pd


# Task 1:
# Read a CSV file into a new Pandas Dataframe object
def csv_to_dataframe(csv_file, col_names=[]):
    if len(col_names) > 0:
        df = pd.read_csv(csv_file, names=col_names)
    else:
        df = pd.read_csv(csv_file)
    return df


# Task 2:
# Takes the integer mean of a column in the input dataframe and fills it in to the column's missing entries
# This method assumes that the column contains integer values and that the missing values are represented as NaN
def impute_missing_vals_in_column_int64(df, column):
    column_mean = np.int64(df[column].mean())
    df[column].fillna(column_mean, inplace=True, downcast='infer')


# Task 3:
# Converts the data in the specified column and maps it to ordinal values as per the dictionary in encode_map
def encode_ordinal(df, column, encode_values):
    df[column].replace(encode_values, inplace=True)


# Task 4:
# One-hot encodes the given column. This function returns a new dataframe with the column onehot-encoded
def encode_onehot(df, column):
    df_dummies = pd.get_dummies(df[column], prefix=column)
    encoded_df = pd.concat([df, df_dummies], axis=1)
    encoded_df.drop([column], axis=1, inplace=True)
    return encoded_df


# Task 5:
# Discretizes the given column into a number of bins of equal width
def discretize_equal_width(df, column, n_bins, labels):
    df[column] = pd.cut(df[column], n_bins, labels=labels)


# Task 5:
# Discretizes the given column into a number of bins containing equal numbers of items
def discretize_equal_freq(df, column, n_bins, labels):
    df[column] = pd.qcut(df[column], n_bins, labels=labels)


# Task 6:
# Calculates the stats for z-score standardization from a training set and applies to a training and test set
def standardize_z_score(training, test, display_mode=True):
    training_mean = np.mean(training)
    training_stdev = np.std(training)
    if display_mode:
        print(f"Training avg= {training_mean}, stdev= {training_stdev}")

    training_z = (training - training_mean) / training_stdev
    test_z = (test - training_mean) / training_stdev
    return training_z, test_z


# Task 7
# Splits a dataframe, using a specified fractional value.
# Returns df_split1, containing (split_frac)% of the entries, and df_split2 with the remainder
def df_split(df, label_column, split_frac):

    # create the first split
    df_split1 = df.sample(frac=split_frac)
    df_split1.sort_index(inplace=True)

    # create the second split by removing the indices of the items from the first
    df_split2 = copy.deepcopy(df)
    df_split2.drop(index=df_split1.index, inplace=True)

    return df_split1, df_split2


# Task 7
# Splits a dataframe, using a specified fractional value. The split will be stratified by class value
# Returns df_split1, containing (split_frac)% of the entries, and df_split2 with the remainder
def df_stratified_split(df, label_column, split_frac):

    # create the training set
    df_split1 = df.groupby(by=label_column).sample(frac=split_frac)
    df_split1.sort_index(inplace=True)

    # create the validation set from the remainder by copying the original dataframe and removing the indices of the
    # sampled items
    df_split2 = copy.deepcopy(df)
    df_split2.drop(index=df_split1.index, inplace=True)

    return df_split1, df_split2


# partitions a dataframe into a specified number of "folds"
# Returns a list of dataframes representing the partitions. Used for cross-fold validation
def df_partition(df, k_folds=5):
    df_copy = copy.deepcopy(df)
    df_partitions = []
    k_fold_index = k_folds

    # take partitions by sampling 1/k-fold index (reducing by 1 every iteration of the loop and removing these from
    # the original df. By reducing the fraction denominator this way the partitions will remain equal.
    for i in range(0, k_folds - 1):
        df_current_kfold = df_copy.sample(frac=1 / k_fold_index)
        k_fold_index -= 1
        df_partitions += [df_current_kfold]
        df_copy.drop(index=df_current_kfold.index, inplace=True)

    # now add what remains in df_copy as the final partition
    df_partitions += [df_copy]

    return df_partitions


# Task 7
# partitions a dataframe into a specified number of "folds", stratified according to the input class label
# Returns a list of dataframes representing the partitions. Used for cross-fold validation of classification data sets
def df_stratified_partition(df, label_column, k_folds=5):
    df_copy = copy.deepcopy(df)
    df_partitions = []
    k_fold_index = k_folds

    # take partitions by sampling 1/k-fold index (reducing by 1 every iteration of the loop and removing these from
    # the original df. By reducing the fraction denominator this way the partitions will remain equal.
    for i in range(0,k_folds-1):
        df_current_kfold = df_copy.groupby(by=label_column).sample(frac=1/k_fold_index)
        k_fold_index -= 1
        df_partitions += [df_current_kfold]
        df_copy.drop(index=df_current_kfold.index, inplace=True)

    # now add what remains in df_copy as the final partition
    df_partitions += [df_copy]

    return df_partitions
