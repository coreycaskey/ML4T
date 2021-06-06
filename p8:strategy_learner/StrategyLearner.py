"""
    Strategy Learner — Q-Learner
"""

import datetime as dt
import pandas as pd
import util as ut
import random

from QLearner import QLearner
from indicators import sma, bbp, macd

#
class StrategyLearner(object):

    #
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ql = QLearner(num_states=1000, num_actions=3, rar=0.0)

    #
    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=10000):
        syms = [symbol]
        dates = pd.date_range(sd - dt.timedelta(days=50), ed) # account for lookback for indicators
        prices = ut.get_data(syms, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print(prices)

        # get indicator dataframes
        indicator_tuple = self.getindicators(prices, symbol, sd)
        sma_df = indicator_tuple[0]
        bbp_df = indicator_tuple[1]
        macd_df = indicator_tuple[2]

        # adjust price dataframes and capture daily returns
        prices = prices[sd:]
        daily_ret, port_val = self.stats(prices, symbol, sv)

        # combine data into overall dataframe
        combined_df = prices.copy()
        combined_df["SMA"] = sma_df[symbol]
        combined_df["BB %"] = bbp_df["Bollinger Band %"]
        combined_df["MACD"] = macd_df["MACD"]
        combined_df["Portfolio"] = port_val
        combined_df["Daily Return"] = daily_ret

        # discretize into bins
        sma_bins = self.discretize(combined_df["SMA"])
        bb_bins = self.discretize(combined_df["BB %"])
        macd_bins = self.discretize(combined_df["MACD"])
        self.state_train = (sma_bins * 100) + (bb_bins * 10) + macd_bins

        # build model
        iterations = 0

        trades = prices.copy()
        trades['Shares'] = 0
        trades.drop([symbol], axis=1, inplace=True)

        while iterations < 400:
            trades_copy = trades.copy()
            net_holdings = 0

            self.ql.querysetstate(self.state_train.iloc[0])

            for date, row in combined_df.iterrows():
                r = combined_df.at[date, "Daily Return"] - (net_holdings * self.impact) # account for impact
                a = self.ql.query(int(self.state_train[date]), r)

                if a == 0: # long
                    if net_holdings == 0: # long
                        trades_copy.at[date, 'Shares'] = 1000
                        net_holdings += 1000

                    elif net_holdings == -1000: # leave position
                        trades_copy.at[date, 'Shares'] = 1000
                        net_holdings += 1000

                elif a == 1: # short
                    if net_holdings == 0: # short
                        trades_copy.at[date, 'Shares'] = -1000
                        net_holdings -= 1000

                    elif net_holdings == 1000: # leave position
                        trades_copy.at[date, 'Shares'] = -1000
                        net_holdings -= 1000

            if iterations > 10 and trades_copy.equals(trades): # converged -> current and previous trades are equal
                break

            trades = trades_copy.copy() # update trades
            iterations += 1

        self.trades = trades

    #
    def test_policy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):
        syms = [symbol]
        dates = pd.date_range(sd - dt.timedelta(days=50), ed) # account for MACD lookback period
        prices = ut.get_data(syms, dates).drop(['SPY'], axis=1)

        if self.verbose:
            print(prices)

        # get indicator dataframes
        indicator_tuple = self.getindicators(prices, symbol, sd)
        sma_df = indicator_tuple[0]
        bbp_df = indicator_tuple[1]
        macd_df = indicator_tuple[2]

        # adjust price dataframes and capture daily returns
        prices = prices[sd:]
        daily_ret, port_val = self.stats(prices, symbol, sv)

        # combine data into overall dataframe
        combined_df = prices.copy()
        combined_df["SMA"] = sma_df[symbol]
        combined_df["BB %"] = bbp_df["Bollinger Band %"]
        combined_df["MACD"] = macd_df["MACD"]
        combined_df["Portfolio"] = port_val
        combined_df["Daily Return"] = daily_ret

        # discretize into bins
        sma_bins = self.discretize(combined_df["SMA"])
        bb_bins = self.discretize(combined_df["BB %"])
        macd_bins = self.discretize(combined_df["MACD"])
        self.state_train = (sma_bins * 100) + (bb_bins * 10) + macd_bins

        # test model
        trades = prices.copy()
        trades['Shares'] = 0
        trades.drop([symbol], axis=1, inplace=True)

        net_holdings = 0

        for date, row in combined_df.iterrows():
            a = self.ql.querysetstate(int(self.state_train[date]))

            if a == 0: # long
                if net_holdings == 0: # long
                    trades.at[date, 'Shares'] = 1000
                    net_holdings += 1000

                elif net_holdings == -1000: # leave position
                    trades.at[date, 'Shares'] = 1000
                    net_holdings += 1000

            elif a == 1: # short
                if net_holdings == 0: # short
                    trades.at[date, 'Shares'] = -1000
                    net_holdings -= 1000

                elif net_holdings == 1000: # leave position
                    trades.at[date, 'Shares'] = -1000
                    net_holdings -= 1000

        return trades

    #
    def discretize(self, indicator):
        return pd.qcut(indicator, 10, labels=False, retbins=True)[0]

    #
    def stats(self, prices, symbol, sv):
        normalize = prices.copy()
        normalize[symbol] = prices[symbol] / float(prices[symbol][0])

        port_val = normalize.sum(axis=1)
        daily_ret = port_val.pct_change(1)

        daily_ret.iloc[0] = 0
        port_val *= sv

        return daily_ret, port_val

    #
    def getindicators(self, prices, symbol, sd):
        sma_tuple = sma(prices, 14, symbol, False)
        bb_tuple = bbp(prices, 14, symbol, False)

        sma_df = sma_tuple[0][sd:]
        bbp_df = bb_tuple[1][sd:]
        macd_df = macd(prices, symbol, False)[sd:]

        return sma_df, bbp_df, macd_df

#
if __name__ == "__main__":
    print("One does not simply think up a strategy")
