# pylint: disable=invalid-name

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.odr import ODR, Model, RealData
from scipy.optimize import minimize
import warnings
from .engle_granger import EngleGrangerPortfolio
from .johansen import JohansenPortfolio


def get_ols_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get OLS hedge ratio: y = beta*X.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    ols_model = LinearRegression(fit_intercept=add_constant)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    exogenous_variables = X.columns.tolist()
    if X.shape[1] == 1:
        X = X.values.reshape(-1, 1)

    y = price_data[dependent_variable].copy()

    ols_model.fit(X, y)
    residuals = y - ols_model.predict(X)

    hedge_ratios = ols_model.coef_
    hedge_ratios_dict = dict(zip([dependent_variable] + exogenous_variables, np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals


def _linear_f_no_constant(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    _, b = beta[0], beta[1:]
    b.shape = (b.shape[0], 1)

    return (x_variable * b).sum(axis=0)


def _linear_f_constant(beta: np.array, x_variable: np.array) -> np.array:
    """
    This is the helper linear model that is used in the Orthogonal Regression.

    :param beta: (np.array) Model beta coefficient.
    :param x_variable: (np.array) Model X vector.
    :return: (np.array) Vector result of equation calculation.
    """

    a, b = beta[0], beta[1:]
    b.shape = (b.shape[0], 1)

    return a + (x_variable * b).sum(axis=0)


def get_tls_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get Total Least Squares (TLS) hedge ratio using Orthogonal Regression.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Hedge ratios dict, X, and y and fit residuals.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    y = price_data[dependent_variable].copy()

    linear = Model(_linear_f_constant) if add_constant is True else Model(_linear_f_no_constant)
    mydata = RealData(X.T, y)
    myodr = ODR(mydata, linear, beta0=np.ones(X.shape[1] + 1))
    res_co = myodr.run()

    hedge_ratios = res_co.beta[1:]  # We don't need constant
    residuals = y - res_co.beta[0] - (X * hedge_ratios).sum(axis=1) if add_constant is True else y - (
            X * hedge_ratios).sum(axis=1)
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals


def _min_hl_function(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Fitness function to minimize in Minimum Half-Life Hedge Ratio algorithm.

    :param beta: (np.array) Array of hedge ratios.
    :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
    :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
    :return: (float) Half-life of mean-reversion.
    """

    spread = y - (beta * X).sum(axis=1)

    return abs(get_half_life_of_mean_reversion(spread))


def get_minimum_hl_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing spread half-life of mean reversion.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y, OLS fit residuals and optimization object.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_hl_function, x0=initial_guess, method='BFGS', tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))
    if result.status != 0:
        warnings.warn('Optimization failed to converge. Please check output hedge ratio! The result can be unstable!')

    return hedge_ratios_dict, X, y, residuals, result


def _min_adf_stat(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Fitness function to minimize in ADF test statistic algorithm.

        :param beta: (np.array) Array of hedge ratios.
        :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
        :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
        :return: (float) Half-life of mean-reversion.
        """

        # Performing Engle-Granger test on spread
        portfolio = EngleGrangerPortfolio()
        spread = y - (beta * X).sum(axis=1)
        portfolio.perform_eg_test(spread)

        return portfolio.adf_statistics.loc['statistic_value'].iloc[0]


    
def get_adf_optimal_hedge_ratio( price_data: pd.DataFrame, dependent_variable: str) -> \
            Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing ADF test statistic.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y, OLS fit residuals and optimization object.
    """

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_adf_stat, x0=initial_guess, method='BFGS', tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))
    if result.status != 0:
        warnings.warn('Optimization failed to converge. Please check output hedge ratio! The result can be unstable!')

    return hedge_ratios_dict, X, y, residuals, result

def get_johansen_hedge_ratio( price_data: pd.DataFrame, dependent_variable: str) -> Tuple[
    dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigenvector.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    # Construct a Johansen portfolio
    port = JohansenPortfolio()
    port.fit(price_data, dependent_variable)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()

    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1.
    hedge_ratios = port.hedge_ratios.iloc[0].to_dict()

    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    # Normalize Johansen cointegration vectors such that dependent variable has a hedge ratio of 1.
    return hedge_ratios, X, y, residuals


def get_half_life_of_mean_reversion(data: pd.Series) -> float:
    """
        Get half-life of mean-reversion under the assumption that data follows the Ornstein-Uhlenbeck process.

        :param data: (np.array) Data points.
        :return: (float) Half-life of mean reversion.
    """

    reg = LinearRegression(fit_intercept=True)

    training_data = data.shift(1).dropna().values.reshape(-1, 1)
    target_values = data.diff().dropna()
    reg.fit(X=training_data, y=target_values)

    half_life = -np.log(2) / reg.coef_[0]

    return half_life


def get_hurst_exponent( data: np.array, max_lags: int = 100) -> float:
    """
        Hurst Exponent Calculation.

        :param data: (np.array) Time Series that is going to be analyzed.
        :param max_lags: (int) Maximum amount of lags to be used calculating tau.
        :return: (float) Hurst exponent.
    """

    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
            for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0

def construct_spread(price_data: pd.DataFrame, hedge_ratios: pd.Series, dependent_variable: str = None) -> pd.Series:
    """
    Construct spread from `price_data` and `hedge_ratios`. If a user sets `dependent_variable` it means that a
    spread will be:

    hedge_ratio_dependent_variable * dependent_variable - sum(hedge_ratios * other variables).
    Otherwise, spread is:  hedge_ratio_0 * variable_0 - sum(hedge ratios * variables[1:]).

    :param price_data: (pd.DataFrame) Asset prices data frame.
    :param hedge_ratios: (pd.Series) Hedge ratios series (index-tickers, values-hedge ratios).
    :param dependent_variable: (str) Dependent variable to use. Set None for dependent variable being equal to 0 column.
    :return: (pd.Series) Spread series.
    """

    weighted_prices = price_data * hedge_ratios  # price * hedge

    if dependent_variable is not None:
        non_dependent_variables = [x for x in weighted_prices.columns if x != dependent_variable]
        return weighted_prices[dependent_variable] - weighted_prices[non_dependent_variables].sum(axis=1)
    else:
        return weighted_prices[weighted_prices.columns[0]] - weighted_prices[weighted_prices.columns[1:]].sum(axis=1)


