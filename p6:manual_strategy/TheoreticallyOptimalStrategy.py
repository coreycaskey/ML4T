"""
    Theoretically Optimal Strategy
"""

import math
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util import get_data, plot_data
from marketsimcode import compute_portvals

#
class TheoreticallyOptimalStrategy(object):

    #
    def __init__(self):
        pass

    #
    def test_policy(self, symbol='AAPL', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        """
            Provides a trading strategy based on looking into the future
        """

        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)

        tos_orders = prices.copy()
        tos_orders[symbol] = np.nan
        tos_orders['Order'] = np.nan
        tos_orders['Shares'] = np.nan

        tos_orders.rename(columns={symbol:'Symbol'}, inplace=True)

        net_holdings = 0 # 0 shares, 1000 shares, or -1000 shares

        for index in range(tos_orders.shape[0] - 1): # iterate over dataframe
            curr_date = tos_orders.index[index]
            next_date = tos_orders.index[index + 1]

            curr_price = prices.at[curr_date, symbol]
            next_price = prices.at[next_date, symbol]

            if curr_price < next_price: # price goes up tomorrow -> BUY shares now

                if net_holdings == 0: # can buy at most 1000 shares
                    tos_orders.loc[curr_date] = pd.Series({'Symbol':symbol, 'Order':'BUY', 'Shares':1000})
                    net_holdings += 1000

                elif net_holdings == -1000: # can buy at most 2000 shares
                    tos_orders.loc[curr_date] = pd.Series({'Symbol':symbol, 'Order':'BUY', 'Shares':2000})
                    net_holdings += 2000

            elif curr_price > next_price: # price goes down tomorrow -> SELL/SHORT shares now

                if net_holdings == 0: # can sell at most 1000 shares
                    tos_orders.loc[curr_date] = pd.Series({'Symbol':symbol, 'Order':'SELL', 'Shares':1000})
                    net_holdings -= 1000

                elif net_holdings == 1000: # can sell at most 2000 shares
                    tos_orders.loc[curr_date] = pd.Series({'Symbol':symbol, 'Order':'SELL', 'Shares':2000})
                    net_holdings -= 2000

        tos_orders.fillna(value=pd.Series({'Symbol':symbol, 'Order':'BUY', 'Shares':0}), axis=0, inplace=True) # replace NaNs with empty BUY orders

        return tos_orders

    #
    def base_policy(self, symbol='AAPL', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
        """
            Implements a basic trading strategy of investing 1000 shares in a stock and holding
            that position for the duration of the time interval
        """

        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates).drop(['SPY'], axis=1)
        actual_start = prices.index[0]

        benchmark_orders = prices.copy()
        benchmark_orders[symbol] = symbol
        benchmark_orders['Order'] = 'BUY'
        benchmark_orders['Shares'] = 0

        benchmark_orders.rename(columns={symbol:'Symbol'}, inplace=True)
        benchmark_orders.at[actual_start, 'Shares'] = 1000

        return benchmark_orders

    #
    def evaluate(self):
        """
            Compares the Theoretically Optimal Strategy approach to the benchmark approach
        """

        sd = dt.datetime(2008,1,1)
        ed = dt.datetime(2009,12,31)

        benchmark_orders = self.base_policy(symbol='JPM', sd=sd, ed=ed)
        benchmark_portval = compute_portvals(benchmark_orders, 100000, 0.0, 0.0) # ignore commission and market impact for theoretically optimal strategy
        benchmark_portval = benchmark_portval.to_frame(name='Benchmark PortVal')
        benchmark_portval /= benchmark_portval.iloc[0]

        tos_orders = self.test_policy(symbol='JPM', sd=sd, ed=ed)
        tos_portval = compute_portvals(tos_orders, 100000, 0.0, 0.0)
        tos_portval = tos_portval.to_frame(name='Theoretically Optimal Strategy PortVal')
        tos_portval /= tos_portval.iloc[0]

        portval_df = pd.concat([benchmark_portval, tos_portval], axis=1)

        portval_graph = portval_df.plot(title="Benchmark & Theoretically Optimal Portfolio Value", fontsize=12, grid=True, color=['blue', 'black'])
        portval_graph.set_xlabel('Date')
        portval_graph.set_ylabel('Normalized Portfolio Value ($)')

        plt.savefig('Figure_4.png')

        benchmark_cr = (benchmark_portval.iloc[-1].at['Benchmark PortVal'] / benchmark_portval.iloc[0].at['Benchmark PortVal']) - 1
        benchmark_adr = benchmark_portval.pct_change(1).mean()['Benchmark PortVal']
        benchmark_sddr = benchmark_portval.pct_change(1).std()['Benchmark PortVal']
        benchmark_sr = math.sqrt(252.0) * (benchmark_adr / benchmark_sddr)

        tos_cr = (tos_portval.iloc[-1].at['Theoretically Optimal Strategy PortVal'] / tos_portval.iloc[0].at['Theoretically Optimal Strategy PortVal']) - 1
        tos_adr = tos_portval.pct_change(1).mean()['Theoretically Optimal Strategy PortVal']
        tos_sddr = tos_portval.pct_change(1).std()['Theoretically Optimal Strategy PortVal']
        tos_sr = math.sqrt(252.0) * (tos_adr / tos_sddr)

        print()
        print('In Sample')

        print('Date Range: {} to {}'.format(sd, ed))

        print('Cumulative Return of Benchmark: {}'.format(benchmark_cr))
        print('Cumulative Return of TOS: {}'.format(tos_cr))

        print('Standard Deviation of Benchmark: {}'.format(benchmark_sddr))
        print('Standard Deviation of TOS: {}'.format(tos_adr))

        print('Average Daily Return of Benchmark: {}'.format(benchmark_adr))
        print('Average Daily Return of TOS: {}'.format(tos_sddr))

        print('Sharpe Ratio of Benchmark: {}'.format(benchmark_sr))
        print('Sharpe Ratio of TOS: {}'.format(tos_sr))
