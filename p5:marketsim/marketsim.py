'''
  Market simulator

  Usage:
  - Point the terminal location to this directory
  - Run the command `PYTHONPATH=../:. python marketsim.py`
'''

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

#
def compute_portvals(orders_file='./orders/orders-01.csv', start_val=1000000, commission=9.95, impact=0.005):
  '''
    This function is tested by the autograder

    NOTE: orders_file may be a string, or it may be a file object. Your code should work correctly with either input
  '''

  orders_df = pd.read_csv(orders_file).sort_values(by=['Date'])

  start_date = orders_df.iloc[0].at['Date']
  end_date = orders_df.iloc[-1].at['Date']
  symbols = list(orders_df['Symbol'].unique())

  prices_df = get_data(symbols, pd.date_range(start_date, end_date)).drop(['SPY'], axis=1)
  prices_df.fillna(method='ffill', inplace=True)
  prices_df.fillna(method='bfill', inplace=True)
  prices_df['Cash'] = 1.0 # add cash column

  trades_df = prices_df.copy() * 0.0
  holdings_df = prices_df.copy() * 0.0

  holdings_df.at[start_date, 'Cash'] = start_val

  for index, row in orders_df.iterrows(): # index for orders_df is an integer
    date = row['Date']
    stock = row['Symbol']
    num_shares = row['Shares']

    stock_price = prices_df.at[date, stock]
    total_price = stock_price * num_shares

    if row['Order'] == 'BUY':
      trades_df.at[date, stock] += num_shares
      transaction_cost = total_price + commission + (total_price * impact)
      trades_df.at[date, 'Cash'] -= transaction_cost

    else: # selling / shorting a stock
      trades_df.at[date, stock] -= num_shares
      transaction_payout = total_price - commission - (total_price * impact)
      trades_df.at[date, 'Cash'] += transaction_payout

  holdings_df.loc[start_date] += trades_df.loc[start_date] # start_date indicates row of index 0

  for i in range(1, trades_df.shape[0]):
    holdings_df.iloc[i] = trades_df.iloc[i] + holdings_df.iloc[i-1]

  value_df = prices_df * holdings_df
  portval_df = value_df.sum(axis=1) # actually a pandas series

  return portval_df

#
def test_code():
  '''
    This function is not called by the autograder

    Any variables defined below will be set to different values by the autograder
  '''

  start_val = 1000000
  portvals = compute_portvals(orders_file='./orders/orders-12.csv', start_val=start_val)

  if not isinstance(portvals, pd.Series):
    print('WARNING: code did not return a Series')
    return

  print('Starting Portfolio Value: {}'.format(start_val))
  print('Final Portfolio Value: {}'.format(portvals.iat[-1]))

#
if __name__ == '__main__':
  '''
    This code is not called by the autograder
  '''

  test_code()
