import copy
import eval
import numpy as np
import pandas as pd


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def classify_cross_validation(df_trn_partitions, model, label_column):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Test the model and record its score
        y_pred = model.fit_predict_classify(df_trn_fold, df_test, label_column)
        y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_classification_score(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), Acc= {score*100:.2f}%')
        scores += [score]

    return scores


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def regression_cross_validation(df_trn_partitions, model, label_column, error):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Test the model and record its score
        y_pred = model.fit_predict_regression(df_trn_fold, df_test, label_column)
        y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_thresh(y_truth, y_pred, error)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), Acc= {score*100:.2f}%')
        scores += [score]

    return scores
