"""
        Market simulator
"""

import pandas as pd
import numpy as np
import datetime as dt
import os

from util import get_data, plot_data

#
def compute_portvals(orders, start_val=1000000, commission=9.95, impact=0.005):
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbols = list(orders['Symbol'].unique())

    prices_df = get_data(symbols, pd.date_range(start_date, end_date)).drop(['SPY'], axis=1)
    prices_df.fillna(method='ffill', inplace=True)
    prices_df.fillna(method='bfill', inplace=True)
    prices_df['Cash'] = 1.0 # add cash column

    trades_df = prices_df.copy() * 0.0
    holdings_df = prices_df.copy() * 0.0

    holdings_df.at[start_date, 'Cash'] = start_val

    for index, row in orders.iterrows():
        date = index
        stock = row['Symbol']
        num_shares = row['Shares']

        stock_price = prices_df.at[date, stock]
        total_price = stock_price * num_shares

        if row['Order'] == 'BUY' and num_shares > 0:
            trades_df.at[date, stock] += num_shares
            transaction_cost = total_price + commission + (total_price * impact)
            trades_df.at[date, 'Cash'] -= transaction_cost

        elif row['Order'] == 'SELL' and num_shares > 0: # selling / shorting a stock
            trades_df.at[date, stock] -= num_shares
            transaction_payout = total_price - commission - (total_price * impact)
            trades_df.at[date, 'Cash'] += transaction_payout

    holdings_df.loc[start_date] += trades_df.loc[start_date] # start_date indicates row of index 0

    for i in range(1, trades_df.shape[0]):
        holdings_df.iloc[i] = trades_df.iloc[i] + holdings_df.iloc[i-1]

    value_df = prices_df * holdings_df
    portval_df = value_df.sum(axis=1) # actually a pandas series

    return portval_df
