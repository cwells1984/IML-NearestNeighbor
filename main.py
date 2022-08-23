# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import dataprep
import knearestneighbor
import numpy as np
import pandas as pd
import preprocessing
import processing


def perform_k_nearest_classification(df, label_column, edit_nn=False, cond_nn=False):

    # First divide the data into training (trn_frac)%, and validation (remaining)% sets
    df_trn, df_val = preprocessing.df_stratified_split(df, label_column, 0.8)

    # If we are editing the dataset points create the edited training set here
    if edit_nn:
        df_edited_trn = knearestneighbor.edit_training_set(df_trn, label_column, False)

    # If we are condensing the dataset points create the condensed training set here
    if cond_nn:
        df_condensed = df_trn.iloc[0:1]
        df_trn_return = copy.deepcopy(df_trn)
        for i in range(3):
            df_trn_return, df_condensed_trn = knearestneighbor.condense_training_set(df_trn_return, df_condensed, label_column, False)

    # Next, divide the training set into k stratified partitions
    df_trn_partitions = preprocessing.df_stratified_partition(df_trn, label_column, 5)

    # Hyperparameter tuning: construct k models each using k-1 of the partitions
    best_model, best_k_param, best_perf = knearestneighbor.tune_classification_model(df_trn_partitions, df_val,
                                                                           knearestneighbor.KNearestNeighbor(), label_column)
    print(f"Best k-param: {best_k_param}")
    print(f"Accuracy for this k-param: {best_perf * 100:.2f}%")

    # Now try with the edited set if we are also performing edited NN
    if edit_nn:
        df_edited_trn_partitions = preprocessing.df_stratified_partition(df_edited_trn, label_column, 5)
        scores = processing.classify_cross_validation(df_edited_trn_partitions, best_model, label_column)
        print(f"Edited Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    elif cond_nn:
        df_cond_trn_partitions = preprocessing.df_stratified_partition(df_condensed_trn, label_column, 5)
        scores = processing.classify_cross_validation(df_cond_trn_partitions, best_model, label_column)
        print(f"Condensed Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
    else:
        # Now that we have found the best model, train on k-1 and test on the last k-fold, getting the average score
        scores = processing.classify_cross_validation(df_trn_partitions, best_model, label_column)
        print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")


def perform_k_nearest_regression(df, label_column, error_threshold):

    # First divide the data into training (trn_frac)%, and validation (remaining)% sets
    df_trn, df_val = preprocessing.df_split(df, label_column, 0.8)

    # Next, divide the training set into k stratified partitions
    df_trn_partitions = preprocessing.df_partition(df_trn, 5)

    # Hyperparameter tuning: construct k models each using k-1 of the partitions
    best_model, best_k_param, best_variance, best_perf = knearestneighbor.tune_regression_model(df_trn_partitions, df_val,
                                                                           knearestneighbor.KNearestNeighbor(), label_column,
                                                                           error_threshold)
    print(f"Best k-param: {best_k_param}")
    print(f"Best variance: {best_variance}")
    print(f"Accuracy for this k-param and variance: {best_perf * 100:.2f}%")

    # Now that we have found the best model, train on k-1 and test on the last k-fold, getting the average score
    scores = processing.regression_cross_validation(df_trn_partitions, best_model, label_column, error_threshold)
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores)*100:.2f}%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Read the datasets for the experiment's execution
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_car = dataprep.prepare_car('datasets/car.data')
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    print("COMPLETED PRE-PROCESSING")
    print("==============================\n")

    # VIDEO PART 1
    # SHOW DATA BEING SPLIT INTO 5 FOLDS
    print("VIDEO PART 1 - BREAST CANCER WISCONSIN PARTITION")
    counts = np.array(df_breast['Class'].value_counts())
    print(f"Original # entries: {len(df_breast)} ({counts[0]}-{counts[1]})")
    df_breast_partitions = preprocessing.df_stratified_partition(df_breast, 'Class', 5)
    i = 1
    for df_breast_partition in df_breast_partitions:
        counts = np.array(df_breast_partition['Class'].value_counts())
        print(f"Partition {i} entries: {len(df_breast_partition)} ({counts[0]}-{counts[1]})")
        i += 1
    print("==============================\n")

    # VIDEO PART 2
    # SHOW CALCULATION OF EUCLIDEAN DISTANCES
    # X = [ 1 1 1 ] , y = [ 0 0 0]
    #     [ 2 2 2 ]
    # RETURNS DISTANCE MATRIX
    # [ 1.73 ]  -> distance of [0 0 0] to [1 1 1]
    # [ 3.46 ]  -> distance of [0 0 0] to [2 2 2]
    print("VIDEO PART 2 - EUCLIDEAN DISTANCE")
    X = np.array([[1, 1, 1], [2, 2, 2]])
    y = np.array([0, 0, 0])
    print(knearestneighbor.euclidean_distances(X, y))
    print("==============================\n")

    # VIDEO PART 3
    # SHOW KERNEL FUNCTION
    # Distance = [ 1.73, 0 ] , Targets = [ 0 2 ]
    # Calculated prediction should be 2
    print("VIDEO PART 3 - KERNEL FUNCTION")
    nearest = np.array([1.73, 0])
    nearest_pred = np.array([0, 2])
    model = knearestneighbor.KNearestNeighbor(variance=1)
    x_pred = model.kernelize(nearest, nearest_pred)
    print("X at [1, 1, 1]")
    print("2 Nearest Points: [0, 0, 0] (pred value 0) and [2, 2, 2] (pred value 2)")
    print(f"Predicted X value = {x_pred}")
    print("==============================\n")

    # VIDEO PART 4
    # SHOW KNN CLASSIFICATION
    print("VIDEO PART 4 - KNN CLASSIFICATION")
    df_simple = pd.read_csv('classification-simple.csv')
    print("Simple data set:")
    print(df_simple)
    print()
    df_simple_trn, df_simple_tst = preprocessing.df_stratified_split(df_simple, 'class', 0.8)
    model = knearestneighbor.KNearestNeighbor(k=3)
    pred = model.fit_predict_classify(df_simple_trn, df_simple_tst, 'class', verbose=True)
    print(f"Predicted value: {pred}")
    print(f"Actual value: {df_simple_tst.iloc[0]['class']}")
    print("==============================\n")

    # VIDEO PART 5
    # SHOW KNN REGRESSION
    print("VIDEO PART 5 - KNN REGRESSION")
    df_simple = pd.read_csv('regression-simple.csv')
    print("Simple data set:")
    print(df_simple)
    print()
    df_simple_trn, df_simple_tst = preprocessing.df_split(df_simple, 'class', 0.8)
    model = knearestneighbor.KNearestNeighbor(k=3, variance=5)
    pred = model.fit_predict_regression(df_simple_trn, df_simple_tst, 'class', verbose=True)
    print(f"Predicted value: {pred}")
    print(f"Actual value: {df_simple_tst.iloc[0]['class']}")
    print("==============================\n")

    # VIDEO PART 6
    # SHOW EDITED KNN
    print("VIDEO PART 6 - EDITED KNN")
    df_car_trn, df_car_tst = preprocessing.df_stratified_split(df_car, 'CAR', 0.8)
    df_car_edited_trn = knearestneighbor.edit_training_set(df_car_trn, 'CAR', True)
    print("==============================\n")

    # Perform K-Nearest Neighbor on Breast Cancer
    print("BREAST CANCER K-NEAREST")
    perform_k_nearest_classification(df_breast, 'Class')
    print("==============================\n")

    # Perform K-Nearest Neighbor on Car
    print("CAR K-NEAREST")
    perform_k_nearest_classification(df_car, 'CAR')
    perform_k_nearest_classification(df_car, 'CAR', edit_nn=True)
    print("==============================\n")

    # Perform K-Nearest Neighbor on Forest Fires
    print("FOREST FIRES K-NEAREST")
    perform_k_nearest_regression(df_forest, 'area', 16)
    print("==============================\n")

    # Perform K-Nearest Neighbor on House Votes
    print("HOUSE VOTES K-NEAREST")
    perform_k_nearest_classification(df_house, 'party')
    print("==============================\n")

    # Perform K-Nearest Neighbor on Machine
    print("MACHINE K-NEAREST")
    perform_k_nearest_regression(df_machine, 'PRP', 40)
    print("==============================\n")

    # Perform K-Nearest Neighbor on Abalone
    print("ABALONE K-NEAREST")
    perform_k_nearest_regression(df_abalone, 'Rings', 1)
    print("==============================\n")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
