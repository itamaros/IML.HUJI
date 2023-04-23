import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = df[df['Temp'] > -50]  # remove illegal or absurd temperature measurements
    df['Year'] = df['Year'].astype(str)  # important for discrete coloring in the graph
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/city_temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df[df['Country'] == 'Israel']
    fig_1 = px.scatter(israel_df,
                       x='DayOfYear', y='Temp',
                       title='Average Daily Temperature as a Function of Day of the Year',
                       color='Year')
    fig_1.update_traces(marker=dict(size=5))
    pio.write_image(fig=fig_1, engine='orca', file='./daily_avg_temp_to_day_of_year_israel.png')

    israel_month_std_df = israel_df.groupby('Month', as_index=False).agg(std=pd.NamedAgg(column='Temp', aggfunc='std'))
    fig_2 = px.bar(israel_month_std_df,
                   x='Month', y='std',
                   title='Standard deviation of temperature per month')
    pio.write_image(fig=fig_2, engine='orca', file='./israel_temp_std_per_month.png')

    # I don't expect the model to succeed equally over all months. You can see That the standard deviation is higher
    # in some months while in others it is quite low. Some months have a low variance (June to September, 6-9),
    # therefore I expect the model to be closer to the real value there, while months like March and April (3,4) have
    # high variance and are more likely to be prone to larger prediction errors.

    # Question 3 - Exploring differences between countries
    month_temp_df = df.groupby(['Country', 'Month'], as_index=False) \
        .agg(avg=pd.NamedAgg(column='Temp', aggfunc='mean'), std=pd.NamedAgg(column='Temp', aggfunc='std'))

    fig_3 = px.line(month_temp_df,
                    x='Month', y='avg',
                    title='Average Temperature per Month',
                    color='Country',
                    error_y='std')
    fig_3.update_layout(yaxis_title='Average Temperature')
    fig_3.show()
    pio.write_image(fig=fig_3, engine='orca', file='./avg_temp_by_country.png')
    """
    Based on this graph, not all countries have the same pattern. The easiest to spot is South Africa, which has an 
    almost "opposite" high and low months of the year's behavior regarding the other countries (This makes sense as 
    Africa's seasons are opposite of Israel's). The model has the highest chance to work on Jordan well (Not surprising,
    it has a similar climate to Israel), as we can see that their graphs are pretty close.
    It might do an OK job on The Netherlands - we can see the distribution is very similar, but the intercept seems 
    different (about 10 deg. difference over all months). And of course, it is easy to see that it will be very bad at 
    predicting the temperature in South Africa, as I previously mentioned.
    """

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df.DayOfYear, israel_df.Temp, 0.75)
    loss_arr = []

    for i, k in enumerate(range(1, 11)):
        fitted = PolynomialFitting(k).fit(train_X.to_numpy(), train_y.to_numpy())
        loss_arr.append([k, np.round(fitted.loss(test_X.to_numpy(), test_y.to_numpy()), 2)])

    loss_arr = pd.DataFrame.from_records(loss_arr, columns=['k', 'MSE loss'])
    print(loss_arr)

    fig_4 = px.bar(loss_arr,
                   x='k', y='MSE loss',
                   title='Loss over different k (highest deg.) values',
                   text='MSE loss')
    pio.write_image(fig=fig_4, engine='orca', file='./israel_k_loss.png')

    """
    Fitted with k=3, we get a loss error of 3.37 - the lowest out of all k values tried.
    out of the simplest (lowest k values, 0-4) models it has the lowest loss, with k=2 at 7.38 loss, and both k=0 and 
    k=4 with significantly higher loss. Therefore we should choose k=3, based on this train-test split
    """

    # Question 5 - Evaluating fitted model on different countries
    israel_fitted = PolynomialFitting(k=3).fit(israel_df.DayOfYear, israel_df.Temp)
    countries = ['South Africa', 'The Netherlands', 'Jordan']

    all_losses = df.groupby('Country').apply(lambda x: np.round(israel_fitted.loss(x.DayOfYear, x.Temp), 2))
    countries_losses = pd.DataFrame({'Country': all_losses.index, 'Loss': all_losses.values})
    countries_losses = countries_losses[countries_losses['Country'].isin(countries)].reset_index(drop=True)

    fig_5 = px.bar(countries_losses,
                   x='Country', y='Loss',
                   text='Loss',
                   title='Loss of Other Countries over Israel-Fitted model',
                   color='Country')
    pio.write_image(fig=fig_5, engine='orca', file='other_countries_loss_using_israel_fitted.png')
    """
    As expected from Q3, The loss over Jordan was the lowest of all 3 countries. Contrary to my prediction,
    the loss over The Netherlands is actually higher than the loss over South Africa. I reckon this is because
    the sum of differences is larger with The Netherlands than with South Africa (almost all months had a 10 deg. diff
    at The Netherlands, not with South Africa).
    Overall, as expected from a model that was trained only on Israel, 
    its predictions were not as accurate over other countries.
    """
