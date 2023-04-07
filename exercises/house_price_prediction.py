from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Only if in train
    if y is not None:
        X.loc[:, 'price'] = y.loc[:]
        # X = X[X['sqft_lot15'] > 0]
        X.loc[:] = X.loc[X['sqft_lot15'] > 0]
        # X.dropna(inplace=True)
        X.drop_duplicates(subset=['id'], inplace=True)

    # FEATURE ENGINEERING:
    # one-hot encoding for zipcodes
    one_hot_zipcode_df = pd.get_dummies(X['zipcode'], prefix='zip')
    X = X.join(one_hot_zipcode_df)
    # age of the house
    X.loc[:, 'date'] = pd.to_datetime(X['date'], format='%Y%m%dT000000', errors='coerce')
    X.loc[:, 'house_age_yr'] = X['date'].dt.year - X['yr_built']
    # square footage ratio
    X.loc[:, 'sqft_ratio'] = X['sqft_living'] / X['sqft_lot']
    # distance from city center
    X.loc[:, 'dist_from_center'] = X.apply(lambda row: dist_from_reference(row['lat'], row['long']), axis=1)
    # bathroom to bedroom ratio
    X.loc[:, 'bath_bed_ratio'] = X['bathrooms'] / X['bedrooms']
    X.loc[:, 'bath_bed_ratio'] = X.loc[:, 'bath_bed_ratio'].replace([np.inf, -np.inf], np.nan)
    X.loc[:, 'bath_bed_ratio'] = X['bath_bed_ratio'].fillna(0)
    # was house renovated
    X.loc[:, 'is_renovated'] = X['yr_renovated'].apply(lambda row: 1 if (row != 0.0) else 0)
    # last column drops
    X.drop(['date', 'id', 'lat', 'long', 'zipcode', 'yr_renovated'], axis=1, inplace=True)

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if y is not None:
        return X.drop('price', axis=1), X.loc[:, 'price']
    return X


def dist_from_reference(lat, long):
    ref_lat, ref_long = 47.612619936344856, -122.20516535787827  # Bellevue Downtown Park
    R = 6371  # radius of the Earth in km
    phi1 = np.radians(lat)
    phi2 = np.radians(ref_lat)
    delta_phi = np.radians(ref_lat - lat)
    delta_lambda = np.radians(ref_long - long)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


def _config_plot(fig, feature, cov):
    fig.update_traces(
    marker=dict(
        size=5,
        symbol="diamond",
        color="#0033cc",
        opacity=0.7,
        line=dict(
            color='white',
            width=0.2
        )
    )
    )
    fig.update_layout(
        title={"text": f"Pearson correlation between {feature} and Response. <br>Pearson Correlation: "
                       f"{np.round(cov, 3)}", "x": 0.5, "y": 0.95},
        template="simple_white",
        font_color="black",
        title_font_family="Helvetica",
        title_font_color="black",
    )


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~(X.columns.str.contains('^zip_', case=False) | X.columns.str.contains('^is_renovated', case=False))]
    for feature in X:
        p_corr = np.cov(X[feature], y)[0][1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x='x', y='y',
                         labels={'x': f'{feature} value', 'y': 'Response values'}, trendline='ols',
                         trendline_color_override='black')
        _config_plot(fig, feature, p_corr)
        # if abs(p_corr) >= 0.4:
        #     print("good: ", feature, np.round(p_corr, 3))
        # if abs(p_corr) <= 0.1:
        #     print("bad: ", feature, np.round(p_corr, 3))
        # fig.write_html(output_path + f"/pearson corr. {feature}" + ".html")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    y = df['price']
    X = df.drop('price', axis=1)
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
