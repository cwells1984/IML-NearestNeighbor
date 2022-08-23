import copy
import datetime
import eval
import numpy as np
import pandas as pd


# Goes through all points in the training set
# If their nearest neighbor's class is NOT equal to theirs, the point is removed
# Returns a condensed training set dataframe and a new training dataframe with the condensed items removed
def condense_training_set(df_trn, df_condensed, label_column, verbose):

    y = df_trn.loc[:, df_trn.columns == label_column].values.ravel()
    df_trn['drop'] = False

    # now for each point in the set
    for i in range(1, len(df_trn)):

        # run KNN for k=1 to predict the point's value
        df_test = df_trn.iloc[i:i+1].loc[:, df_trn.columns != 'drop']
        model = KNearestNeighbor(k=1)
        pred = model.fit_predict_classify(df_condensed, df_test, label_column)[0]

        # if the prediction is accurate add this point to the condensed set
        if pred == y[i]:
            df_condensed = pd.concat([df_condensed, df_test])
            df_trn.iloc[i]['drop'] = True

    if verbose:
        print(f"original df rows={len(df_trn)}")
        print(f"condensed df rows={len(df_condensed)}")

    df_trn_return = df_trn.loc[df_trn['drop'] == False, df_trn.columns != 'drop']
    return df_trn_return, df_condensed


# Goes through all points in the training set
# If their nearest neighbor's class is NOT equal to theirs, the point is removed
# Returns an edited training set dataframe
def edit_training_set(df_trn, label_column, verbose):
    df_trn_copy = copy.deepcopy(df_trn)
    df_trn_copy.reset_index(inplace=True)
    X = df_trn.loc[:, df_trn.columns != label_column].values
    y = df_trn.loc[:, df_trn.columns == label_column].values.ravel()
    nn_to_remove = set()

    # For each item in the training set...
    for i in range(len(df_trn)):

        # Record the class and calculate the distances to all other points in the training set
        current_class = df_trn_copy.iloc[i][label_column]
        distances = euclidean_distances(X, X[i])
        df_trn_copy['distances'] = distances
        df_trn_copy.sort_values(by='distances', inplace=True)

        # Since the shortest distance will always be to itself, the 2nd shortest is to its nearest neighbor
        nn_class = df_trn_copy.iloc[1][label_column]
        df_trn_copy.drop(columns=['distances'], inplace=True)

        # If the classes don't match add the index to the set of indices to drop
        if current_class != nn_class:
            nn_to_remove.add(i)

    # Drop the row indices marked for deletion
    if verbose:
        print(f"original df rows={len(df_trn_copy)}")
    df_trn_copy.drop(list(nn_to_remove), inplace=True)
    if verbose:
        print(f"new df rows={len(df_trn_copy)}")

    return df_trn_copy


# Given an MxN matrix X and an Mx1 vector y, produces an N-length array containing the distance of each value in
# the matrix from y
def euclidean_distances(X, y):

    # get the delta of the point to all the training data points
    delta = copy.deepcopy(X)
    delta = np.power(np.abs(X - y), 2)

    # now get the distance from all the deltas
    distances = np.power(np.sum(delta, axis=1, keepdims=True), 1 / 2)
    return distances


# Hyperparameter tuning: construct k models each using k-1 of the partitions and verify against the validation set
# Returns the model with the best parameters
def tune_classification_model(df_trn_partitions, df_val, model_to_use, label_column):
    k_folds = len(df_trn_partitions)
    best_model = None
    best_model_perf = 0
    k_parameter = 5

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_tuning = pd.DataFrame(columns=df_trn_partitions[0].columns)
        for j in range(k_folds):
            if i != j:
                df_tuning = pd.concat([df_tuning, df_trn_partitions[j]])

        # Test the model and designate it the best if it outperforms the current best
        model = copy.deepcopy(model_to_use)
        y_pred = model.fit_predict_classify(df_tuning, df_val, label_column)
        y_truth = df_val[label_column].values.ravel()
        score = eval.eval_classification_score(y_truth, y_pred)
        if score > best_model_perf:
            best_model = model
            best_k_param = k_parameter
            best_model_perf = score
        k_parameter += 5

    return best_model, best_k_param, best_model_perf


# Hyperparameter tuning: construct k models each using k-1 of the partitions and verify them against the validation set
# Returns the model with the best performance
def tune_regression_model(df_trn_partitions, df_val, model_to_use, label_column, error):
    k_folds = len(df_trn_partitions)
    best_model = None
    best_k_param = 0
    best_variance = 1
    best_model_perf = 0
    k_parameter = 5
    variance = 5

    # Tune k for a midrange variance
    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_tuning = pd.DataFrame(columns=df_trn_partitions[0].columns)
        for j in range(k_folds):
            if i != j:
                df_tuning = pd.concat([df_tuning, df_trn_partitions[j]])

        # Test the model and designate it the best if it outperforms the current best
        model = copy.deepcopy(model_to_use)
        model.k = k_parameter
        model.variance = variance

        y_pred = model.fit_predict_regression(df_tuning, df_val, label_column)
        y_truth = df_val[label_column].values.ravel()
        score = eval.eval_thresh(y_truth, y_pred, error)
        if score > best_model_perf:
            best_model = model
            best_k_param = k_parameter
            best_variance = variance
            best_model_perf = score
        k_parameter += 5

    # now tune the variance using the best k discovered above
    variance = 1
    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_tuning = pd.DataFrame(columns=df_trn_partitions[0].columns)
        for j in range(k_folds):
            if i != j:
                df_tuning = pd.concat([df_tuning, df_trn_partitions[j]])

        # Test the model and designate it the best if it outperforms the current best
        model = copy.deepcopy(model_to_use)
        model.k = best_k_param
        model.variance = variance

        y_pred = model.fit_predict_regression(df_tuning, df_val, label_column)
        y_truth = df_val[label_column].values.ravel()
        score = eval.eval_thresh(y_truth, y_pred, error)
        if score > best_model_perf:
            best_model = model
            best_k_param = best_k_param
            best_variance = variance
            best_model_perf = score
        variance += 2

    return best_model, best_k_param, best_variance, best_model_perf


class KNearestNeighbor:

    k = 1
    variance = 1

    def __init__(self, k=1, variance=1):
        self.k = k
        self.variance = variance

    def fit_predict_regression(self, df_trn, df_tst, label_column, verbose=False):
        X_trn = df_trn.loc[:, df_trn.columns != label_column].values
        y_trn = df_trn.loc[:, df_trn.columns == label_column].values.ravel()
        X_tst = df_tst.loc[:, df_tst.columns != label_column].values
        y_tst = df_tst.loc[:, df_tst.columns == label_column].values.ravel()
        y_pred = []
        n_dim = np.shape(X_trn)[1]

        # For each point in the test data...
        for i in range(len(df_tst)):
            start_time = datetime.datetime.now()

            # ... Get the distances to all points in the training set
            distances = euclidean_distances(X_trn, X_tst[i])

            # now arrange as a dataframe and sort
            df_distances = copy.deepcopy(df_trn)
            # df_distances = df_distances.loc[:, df_trn.columns == label_column]
            df_distances['distance'] = distances

            # sort the list by distance, ascending
            df_distances.sort_values(by='distance', inplace=True)

            # reduce the list to the k nearest values
            df_distances = df_distances.iloc[0:self.k]
            if verbose:
                print(f"From point at position {df_tst.iloc[i]['position']}")
                print(f"Nearest {self.k} points:")
                print(df_distances)

            # kernelize based on the nearest neighbors
            X_nearest = df_distances.loc[:, df_distances.columns == 'distance'].values.ravel()
            y_nearest = df_distances.loc[:, df_distances.columns == label_column].values.ravel()

            # apply a gaussian rbf kernel to get the predicted value
            y_pred += [self.kernelize(X_nearest, y_nearest)]

            # See how long
            end_time = datetime.datetime.now()
            # print(f"Loop {i} of {len(df_tst)}: {end_time - start_time}")

        # return predicted scores
        return y_pred

    def fit_predict_classify(self, df_trn, df_tst, label_column, verbose=False):
        X_trn = df_trn.loc[:, df_trn.columns != label_column].values
        y_trn = df_trn.loc[:, df_trn.columns == label_column].values.ravel()
        X_tst = df_tst.loc[:, df_tst.columns != label_column].values
        y_tst = df_tst.loc[:, df_tst.columns == label_column].values.ravel()
        y_pred = []
        n_dim = np.shape(X_trn)[1]

        # For each point in the test data...
        for i in range(len(df_tst)):
            start_time = datetime.datetime.now()

            # ... Get the distances to all points in the training set
            distances = euclidean_distances(X_trn, X_tst[i])

            # now arrange as a dataframe and sort
            df_distances = copy.deepcopy(df_trn)
            df_distances['distance'] = distances

            # sort the list by distance, ascending
            df_distances.sort_values(by='distance', inplace=True)

            # reduce the list to the k nearest values
            df_distances = df_distances.iloc[0:self.k]
            if verbose:
                print(f"From point at position {df_tst.iloc[i]['position']}")
                print(f"Nearest {self.k} points:")
                print(df_distances)

            # predicted value is the most frequently occurring class in the list
            y_pred += [df_distances[label_column].value_counts().index[0]]

            # See how long
            end_time = datetime.datetime.now()
            # print(f"Loop {i} of {len(df_tst)}: {end_time - start_time}")

        # return predicted scores
        return y_pred

    # Produces a kernelized predicted value (Gaussian RBF)
    def kernelize(self, distances, nearest_pred):
        best_weight = 0
        point_pred = 0

        for i in range(len(distances)):
            weight = np.exp((-1 * distances[i])/self.variance)
            if weight > best_weight:
                best_weight = weight
                point_pred = weight * nearest_pred[i]

        return point_pred
