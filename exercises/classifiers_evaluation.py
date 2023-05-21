from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset('../datasets/' + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron(callback=lambda per, _, __: losses.append(per.loss(X, y)))
        p.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            marker=dict(color='black')
        ))
        fig.update_layout(
            title={
                'text': f'Perceptron Loss in Training - <i>{n}</i> dataset',
                'x': 0.5,
                'y': 0.95
            },
            template="simple_white",
            font_color="black",
            title_font_family="Helvetica",
            title_font_color="black",
            xaxis_title='Training Iteration number',
            yaxis_title='Misclassification Error (normalized)',
        )
        fig.write_html('perceptron fit ' + n + ' losses.html')


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset('../datasets/' + f)

        # Fit models and predict over training set
        gnb_model, lda_model = GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)
        gnb_pred, lda_pred = gnb_model.predict(X), lda_model.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        subs = make_subplots(rows=1, cols=2,
                             subplot_titles=(
                                 f"Gaussian Naive Bayes (accuracy: {np.round(accuracy(y, gnb_pred) * 100, 2)}%)",
                                 f"LDA (accuracy: {np.round(accuracy(y, lda_pred) * 100, 2)}%)"
                             ))

        # Add traces for data-points setting symbols and colors
        subs.add_traces([
            go.Scatter(x=X[:, 0], y=X[:, 1],
                       mode='markers',
                       marker=dict(color=gnb_pred, symbol=class_symbols[y], colorscale=class_colors(3), size=10,
                                   opacity=0.7)),
            go.Scatter(x=X[:, 0], y=X[:, 1],
                       mode='markers',
                       marker=dict(color=lda_pred, symbol=class_symbols[y], colorscale=class_colors(3), size=10,
                                   opacity=0.7))
        ], rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        subs.add_traces([
            go.Scatter(x=gnb_model.mu_[:, 0], y=gnb_model.mu_[:, 1],
                       mode='markers',
                       marker=dict(symbol='x', color='black', size=10)),
            go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1],
                       mode='markers',
                       marker=dict(symbol='x', color='black', size=10))
        ], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gnb_model.mu_)):
            subs.add_traces([
                get_ellipse(gnb_model.mu_[i], np.diag(gnb_model.vars_[i])),
                get_ellipse(lda_model.mu_[i], lda_model.cov_)
            ], rows=[1, 1], cols=[1, 2])

        subs.update_yaxes(scaleanchor='x', scaleratio=1)
        subs.update_layout(
            title={
                'text': f'<b>Comparison of Naive Gaussian and LDA estimators on {f[:-4]} dataset</b>',
                'x': 0.5
            },
            showlegend=False
        )

        # save plots
        pio.write_html(subs, file=f"naive_bayes_vs_lda_{f[:-4]}.html")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
