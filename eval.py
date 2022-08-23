import numpy as np

# Returns the classification score for a specified ground truth - predicted value pair
def eval_classification_score(y_truth, y_pred):
    y_corr = 0
    num_y = len(y_truth)

    # calculates the fraction of correct predictions
    for i in range(num_y):
        if y_truth[i] == y_pred[i]:
            y_corr += 1

    return y_corr / num_y


# Returns the mean squared error for a specified ground truth - predicted value pair
def eval_mse(y_truth, y_pred):
    y_total_error = 0
    num_y = len(y_truth)

    # sums the squares of the errors
    for i in range(num_y):
        y_total_error += (y_pred[i] - y_truth[i]) ** 2

    return (y_total_error / num_y)


# Returns whether the error for a specified ground truth is within a predicted threshold
def eval_thresh(y_truth, y_pred, thresh):
    y_corr = 0
    num_y = len(y_truth)

    # calculates the fraction of correct predictions
    for i in range(num_y):
        if np.abs(y_truth[i] - y_pred[i]) < thresh:
            y_corr += 1

    return y_corr / num_y