from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var = 10, 1
    samples = np.random.normal(loc=mu, scale=var, size=1000)

    estimator = UnivariateGaussian().fit(samples)
    print("({}, {})".format(np.round(estimator.mu_, 3), np.round(estimator.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    expectations = [abs(mu - UnivariateGaussian().fit(samples[:n]).mu_) for n in range(10, 1010, 10)]
    fig_1 = px.scatter(x=[n for n in range(10, 1010, 10)], y=expectations,
                       labels={
                           "x": "sample size",
                           "y": "absolute distance"})

    fig_1.update_layout(
        title={
            "text": "Error of Estimated Value of Expectation, as a Function of Sample Size",
            "x": 0.5,
            "y": 0.95},
        template="simple_white",
        font_color="black",
        title_font_family="Helvetica",
        title_font_color="black",
    )
    fig_1.update_traces(
        marker=dict(size=5, symbol="diamond", color="black"),
        selector=dict(mode="markers"),
    )
    fig_1.update_xaxes(
        title="Sample Size",
        title_font_family="Times New Roman", tickmode="linear", tick0=0, dtick=50)
    fig_1.update_yaxes(
        title="Absolute distance between est. and true value of expectation",
        title_font_family="Times New Roman")
    fig_1.write_html("Q2.html")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = estimator.pdf(samples)
    X = np.column_stack((samples, pdfs))
    X = X[np.argsort(X[:, 1])]

    fig_2 = px.scatter(x=X[:, 0], y=X[:, 1],
                       labels={
                           "x": "sample",
                           "y": "empirical PDF"})
    fig_2.update_layout(
        title={
            "text": "Empirical PDF Under the Fitted Model",
            "x": 0.5,
            "y": 0.95},
        template="simple_white",
        font_color="black",
        title_font_family="Helvetica",
        title_font_color="black")
    fig_2.update_traces(
        marker=dict(size=5, symbol="diamond", color="black"),
        selector=dict(mode="markers"))
    fig_2.update_xaxes(
        title="x (sample)",
        title_font_family="Times New Roman")
    fig_2.update_yaxes(
        title="Empirical PDF",
        title_font_family="Times New Roman")
    fig_2.write_html("Q3.html")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov = np.array(
        [[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean=[0, 0, 4, 0], cov=cov, size=1000)
    estimator = MultivariateGaussian().fit(samples)
    pdfs = estimator.pdf(samples)
    print("Estimated expectation:", np.round(estimator.mu_, 3), sep="\n", end="\n\n")
    print("Estimated covariance matrix:", np.round(estimator.cov_, 3), sep="\n", end="\n\n")

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, 200)
    m = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            m[i, j] = estimator.log_likelihood([f_values[i], 0, f_values[j], 0], cov, samples)

    heatmap = go.Figure(go.Heatmap(x=f_values, y=f_values, z=m), layout={
        "template": "simple_white",
        "font_color": "black",
        "title_font_family": "Times New Roman",
        "title_font_color": "black",
        "title": "Empirical Log-Likelihood of Multivariate Gaussian Estimation as a Function of Expectation features 1, 3",
        "xaxis_title": "f1",
        "yaxis_title": "f3",
    }
                        )
    heatmap.update_layout(
        xaxis={"tickmode": "linear", "tick0": "0", "dtick": "1"},
        yaxis={"tickmode": "linear", "tick0": "0", "dtick": "2"}
    )
    heatmap.write_html("Q5.html")

    # Question 6 - Maximum likelihood
    argmax_row, argmax_col = np.unravel_index(m.argmax(), m.shape)
    print("The model that achieved maximal log-likelihood value (upon examining entries 1, 3 of the Expectation): ")
    print("[{:.3f}, {:.3f}] achieved a value of {:.3f}".format(f_values[argmax_row], f_values[argmax_col], m.max()))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
