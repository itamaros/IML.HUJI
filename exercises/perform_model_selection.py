from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    # X_train, y_train, X_test, y_test = tts(X, y, test_size=n_samples, random_state=0)
    X_train, y_train, X_test, y_test = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    cv_number = 5
    ridge_lambdas = np.linspace(0, 1, n_evaluations)
    lasso_lambdas = np.linspace(0.001, 2, n_evaluations)
    ridge_scores = np.zeros((n_evaluations, 2))
    lasso_scores = np.zeros((n_evaluations, 2))

    for i, (ridge_lam, lasso_lam) in enumerate(zip(ridge_lambdas, lasso_lambdas)):
        ridge = RidgeRegression(ridge_lam)
        lasso = Lasso(lasso_lam, max_iter=5000)
        ridge_scores[i] = cross_validate(ridge, X_train, y_train, mean_square_error, cv=cv_number)
        lasso_scores[i] = cross_validate(lasso, X_train, y_train, mean_square_error, cv=cv_number)

    # plot results
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Ridge Regression', 'Lasso Regression'), shared_xaxes=True) \
        .add_traces([go.Scatter(x=ridge_lambdas, y=ridge_scores[:, 0], name='Ridge Train Error'),
                     go.Scatter(x=ridge_lambdas, y=ridge_scores[:, 1], name='Ridge Validation Error'),
                     go.Scatter(x=lasso_lambdas, y=lasso_scores[:, 0], name='Lasso Train Error'),
                     go.Scatter(x=lasso_lambdas, y=lasso_scores[:, 1], name='Lasso Validation Error')],
                    rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    fig.update_layout(title=f'Train and Validation Error Average over {cv_number} folds', width=1500, height=750) \
        .update_xaxes(title='Regularization Parameter')
    fig.write_html('ridge_lasso_regression_cv_errors.html')

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lam = ridge_lambdas[np.argmin(ridge_scores[:, 1])]
    best_lasso_lam = lasso_lambdas[np.argmin(lasso_scores[:, 1])]

    ls_loss = LinearRegression().fit(X_train, y_train).loss(X_test, y_test)
    ridge_loss = RidgeRegression(best_ridge_lam).fit(X_train, y_train).loss(X_test, y_test)
    lasso_loss = mean_square_error(y_test, Lasso(alpha=best_lasso_lam).fit(X_train, y_train).predict(X_test))
    print('Best regularization lambdas:')
    print(f'\tRidge: {np.round(best_ridge_lam, 5)}')
    print(f'\tLasso: {np.round(best_lasso_lam, 5)}')
    print(f'Test errors over test set:')
    print(f'\tLest Squares: {np.round(ls_loss, 2)}')
    print(f'\tRidge: {np.round(ridge_loss, 2)}')
    print(f'\tLasso: {np.round(lasso_loss, 2)}')


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
