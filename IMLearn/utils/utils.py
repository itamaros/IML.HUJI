from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    n = int(np.ceil(len(X) * train_proportion))
    index_shuffle = np.random.permutation(len(X))
    shuffled_X = X.iloc[index_shuffle]
    shuffled_y = y.iloc[index_shuffle]
    return shuffled_X.iloc[:n], shuffled_y.iloc[:n], shuffled_X.iloc[n:], shuffled_y.iloc[n:]


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()


if __name__ == '__main__':  # TODO remove
    np.random.seed(0)
    # X = np.arange(1, 10, 1).reshape(3, 3)
    X = pd.DataFrame(np.arange(1, 100, 1).reshape(33, 3))
    y = pd.Series(np.arange(1, 100, 1)).transpose()
    a, b, c, d = split_train_test(X, y)
    print("train X:\n", a.shape, end="\n\n")
    print("train y:\n", b.shape, end="\n\n")
    print("test X:\n", c.shape, end="\n\n")
    print("test y:\n", d.shape, end="\n\n")
