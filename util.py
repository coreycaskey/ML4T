'''
  Utility Functions
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

#
def get_data(symbols, dates, add_spy=True, col_name='Adj Close'):
  '''
    Read stock data (adjusted close) for given symbols from CSV files
  '''

  df = pd.DataFrame(index=dates)

  if add_spy and 'SPY' not in symbols:
    symbols = ['SPY'] + list(symbols) # handles case where symbols is an numpy array of 'object'

  for symbol in symbols:
    df_temp = pd.read_csv(
      symbol_to_path(symbol),
      index_col='Date',
      parse_dates=True,
      usecols=['Date', col_name],
      na_values=['nan'],
    )
    df_temp = df_temp.rename(columns={ col_name:symbol })
    df = df.join(df_temp) # join dataframes on date values (adds NaN for date mismatches)

    if symbol == 'SPY': # drop dates SPY did not trade
      df = df.dropna(subset=['SPY'])

  return df

#
def symbol_to_path(symbol, base_dir=None):
  '''
    Return CSV file path for given ticker symbol
  '''

  if base_dir is None:
    base_dir = os.environ.get('MARKET_DATA_DIR', '../data/') # pull path from env key or use default path

  return os.path.join(base_dir, '{}.csv'.format(str(symbol)))

#
def plot_data(df, title='Stock prices', xlabel='Date', ylabel='Price'):
  '''
    Plot stock prices with a custom title and meaningful axis labels.
  '''

  ax = df.plot(title=title, fontsize=12)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.show()


def get_orders_data_file(basefilename):
    return open(
        os.path.join(
            os.environ.get('ORDERS_DATA_DIR', 'orders/'), basefilename
        )
    )


def get_learner_data_file(basefilename):
    return open(
        os.path.join(
            os.environ.get('LEARNER_DATA_DIR', 'Data/'), basefilename
        ),
        'r',
    )


def get_robot_world_file(basefilename):
    return open(
        os.path.join(
            os.environ.get('ROBOT_WORLDS_DIR', 'testworlds/'), basefilename
        )
    )
