"""
    Optimize a Portfolio

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python optimization.py`
"""

from util import get_data

import datetime as dt
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo

#
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    """
        This function is tested by the autograder
    """

    dates = pd.date_range(sd, ed)
    portfolio_and_spy_prices = get_data(syms, dates) # read in adjusted closing prices for given symbols
    portfolio_prices = portfolio_and_spy_prices[syms]
    spy_prices = portfolio_and_spy_prices['SPY']

    min_sharpe_ratio = spo.minimize(
        get_sharpe_ratio,
        np.asarray([1.0 / len(syms)] * len(syms)),
        args=(syms, portfolio_prices),
        method='SLSQP',
        bounds=[(0, 1)] * len(syms),
        constraints=[{ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) }]
    )

    optimal_allocations = min_sharpe_ratio['x']
    portfolio_statistics = assess_portfolio(optimal_allocations, syms, portfolio_prices)

    cumulative_return, average_daily_return, std_daily_return, sharpe_ratio, portfolio_values = \
        portfolio_statistics[0], portfolio_statistics[1], portfolio_statistics[2], portfolio_statistics[3], portfolio_statistics[4]

    if gen_plot:
        normalized_spy_prices = spy_prices.copy()
        normalized_spy_prices /= normalized_spy_prices.iloc[0]

        df = pd.concat([portfolio_values, normalized_spy_prices], keys=['Portfolio', 'SPY'], axis=1)
        save_plot(df)

    return optimal_allocations, cumulative_return, average_daily_return, std_daily_return, sharpe_ratio

#
def save_plot(df):
    ax = df.plot(title='Daily Portfolio Value and SPY', fontsize=12, grid=True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.savefig('Figure_1.png')

#
def get_sharpe_ratio(x, symbols, portfolio_prices):
    """
        Retrieves the (negated) Sharpe Ratio for a portfolio, where x
        is an array of potential stock allocations
    """

    return assess_portfolio(x, symbols, portfolio_prices, False) * -1

#
def assess_portfolio(allocations, symbols, portfolio_prices, multi_val_return=True):
    """
        Calculates critical statistics for a given portfolio that's using the
        provided stock symbols and allocations
    """

    normalized_prices = portfolio_prices.copy()
    normalized_prices /= normalized_prices.iloc[0]
    alloc_df = normalized_prices * allocations

    # since the initial investment is a constant, we can
    # exclude that step and reach the same result

    portfolio_values = alloc_df.sum(axis=1)
    cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    daily_returns = (portfolio_values.pct_change()).iloc[1:]
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    sharpe_ratio = math.sqrt(252) * (avg_daily_return / std_daily_return) # sharpe ratio for daily historical data

    if multi_val_return:
        return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_values
    else:
        return sharpe_ratio

#
def test_code():
    """
        This function is not called by the autograder

        Any variables defined below will be set to different values by the autograder
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    print('Start Date:', start_date)
    print('End Date:', end_date)
    print('Symbols:', symbols)
    print('Allocations:', allocations)
    print('Sharpe Ratio:', sr)
    print('Volatility (stdev of daily returns):', sddr)
    print('Average Daily Return:', adr)
    print('Cumulative Return:', cr)

#
if __name__ == '__main__':
    """
        This code is not called by the autograder
    """

    test_code()
