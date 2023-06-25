import numpy as np
import pandas as pd
import itertools
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}",
                                      width=750, height=750))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_lst = []
    deltas = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        weights_lst.append(weights)
        deltas.append(delta)

    return callback, values, weights_lst, deltas


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    norm_dict = {L1: "l1", L2: "l2"}
    losses = {"l1": [], "l2": []}

    for eta, l in itertools.product(etas, [L1, L2]):
        callback, values, weights_lst, deltas = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=FixedLR(eta), callback=callback).fit(l(init), None, None)

        # Q1 - Plotting descent paths for each setting:
        trajectory_fig = plot_descent_path(l, np.array(weights_lst), title=f"Norm = {norm_dict[l]}, Step(η) = {eta}.")
        trajectory_fig.write_html(f'GD_{norm_dict[l]}_{eta}.html')

        # Q3 - Plotting the convergence rate for each setting:
        norm_fig = go.Figure(go.Scatter(x=np.linspace(0, len(deltas) - 1, len(deltas)),
                                        y=deltas, mode="lines+markers"))
        norm_fig.update_layout(title=f"Learning as function of GD iteration with.<br>step (η)={eta}, norm={norm_dict[l]}.",
                               width=500, height=500)
        norm_fig.write_html(f'GD_{norm_dict[l]}_{eta}_norm.html')

        # Q4 - Accumulating losses for each setting:
        losses[norm_dict[l]].append(min(values))

    # Q4 - Printing the lowest loss achieved for each setting:
    print(f"The lowest loss achieved when minimizing L1 is: ", np.round(min(losses["l1"]), 9))
    print(f"The lowest loss achieved when minimizing L2 is: ", np.round(min(losses["l2"]), 9))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    l1_fig = go.Figure()
    for gamma in gammas:
        callback, values, weights, deltas = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback).fit(L1(init), None, None)
        l1_fig.add_trace(go.Scatter(x=np.linspace(0, len(deltas) - 1, len(deltas)),
                                    y=deltas, mode="lines+markers", name=gamma))
        # Q6 - Lowest loss achieved using the exponential decay function:
        print(f"The lowest loss achieved when minimizing L1 using gamma={gamma} is: ", min(values))

    # Q5 - Plot algorithm's convergence for the different values of gamma
    l1_fig.update_layout(title=f"Convergence rate for all decay rates (γ).<br>step (η)={eta}, norm=L1.",
                         legend_title_text='γ:',
                         template="plotly_white")
    l1_fig.write_html(f'Q5 - GD_L1_{eta}_exp_decay.html')

    # Q7 - Plot descent path for gamma=0.95
    callback, values, weights, norms = get_gd_state_recorder_callback()
    GradientDescent(ExponentialLR(eta, gammas[1]), callback=callback).fit(L1(init), None, None)
    trajectory_fig = plot_descent_path(L1, np.array(weights), title=f"Norm = L1, Step(η) = {eta}, gamma=0.95.")
    trajectory_fig.write_html(f'GD_L1_{eta}_exp_decay_gamma_0.95.html')


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Q8 - Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression().fit(np.array(X_train), np.array(y_train))
    y_prob = model.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_alpha = thresholds[np.argmax(tpr - fpr)]

    roc = go.Figure(data=[
        go.Scatter(x=[0, 1], y=[0, 1],
                   mode='lines',
                   line=dict(color='black', dash='dash'),
                   name='Random Class Assignment'),
        go.Scatter(x=fpr, y=tpr,
                   mode='markers+lines',
                   text=thresholds, name='', showlegend=False,
                   hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")])
    roc.update_layout(
        title=f"ROC curve for SA heart disease data.<br>Best alpha={np.round(best_alpha, 3)}, "
              f"Train AUC={np.round(auc(fpr, tpr), 3)}",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        template="plotly_white"
    )
    roc.write_html(f'heart_disease_roc.html')

    # Q9
    print("Q9 - The best α is: ", np.round(best_alpha, 3))
    print("Q9 - The model's test error is: ", np.round(model.loss(np.array(X_test), np.array(y_test)), 3))

    # Q10 - Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    alpha = 0.5
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for norm in ["l1", "l2"]:
        train_score_lst, val_score_lst = [], []
        for lam in lambdas:
            lam_model = LogisticRegression(penalty=norm, lam=lam, alpha=alpha,
                                           solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))
            avg_train_score, avg_val_score = cross_validate(lam_model, np.array(X_train), np.array(y_train),
                                                            misclassification_error)
            train_score_lst.append(avg_train_score)
            val_score_lst.append(avg_val_score)

        best_lambda = lambdas[np.argmin(val_score_lst)]
        best_lam_model = LogisticRegression(penalty=norm, lam=best_lambda, alpha=alpha,
                                            solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))\
            .fit(np.array(X_train), np.array(y_train))
        test_error = best_lam_model.loss(np.array(X_test), np.array(y_test))
        print(f"Q10 - The best λ for {norm}-regularized logistic regression is: ", np.round(best_lambda, 3),
              " and its test error is: ", np.round(test_error, 3))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
