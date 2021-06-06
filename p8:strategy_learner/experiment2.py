"""
    Experiment 1

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python experiment2.py`
"""

import math
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import sma, bbp, macd
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

#
def compare(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), impact=0.0):
    sl = StrategyLearner(impact=impact)
    sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)

    trades = sl.test_policy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    trades['Symbol'] = symbol
    trades['Order'] = 'BUY'

    strategy_trades = 0

    for date, row in trades.iterrows():
        if trades.at[date, 'Shares'] == -1000:
            trades.at[date, 'Order'] = 'SELL'

        if row['Shares'] != 0:
            strategy_trades += 1

    strategy = compute_portvals(trades, start_val=100000, commission=0.0, impact=impact)

    strategy_portval = strategy.to_frame(name='Strategy Learner PortVal')
    strategy_portval /= strategy_portval.iloc[0]

    return strategy_portval, strategy_trades

#
if __name__ == "__main__":
    impact = 0.002
    chart_dfs = []

    print("In Sample")
    print("Date Range: {} to {}".format(dt.datetime(2008,1,1), dt.datetime(2009,12,31)))

    for i in range(5):
        strategy_portval, strategy_trades = compare(impact=impact)

        # strategy learner values
        strategy_cr = (strategy_portval.iloc[-1].at['Strategy Learner PortVal'] / strategy_portval.iloc[0].at['Strategy Learner PortVal']) - 1
        strategy_adr = strategy_portval.pct_change(1).mean()['Strategy Learner PortVal']
        strategy_sddr = strategy_portval.pct_change(1).std()['Strategy Learner PortVal']
        strategy_sr = math.sqrt(252.0) * (strategy_adr / float(strategy_sddr))

        print("Cumulative Return of Strategy (Impact: " + str(impact) + "): {}".format(strategy_cr))
        print("Standard Deviation of Strategy (Impact: " + str(impact) + "): {}".format(strategy_sddr))
        print("Average Daily Return of Strategy (Impact: " + str(impact) + "): {}".format(strategy_adr))
        print("Sharpe Ratio of Strategy (Impact: " + str(impact) + "): {}".format(strategy_sr))
        print("Number of Trades for Strategy (Impact: " + str(impact) + "): {}".format(strategy_trades))

        strategy_portval.rename(columns={"Strategy Learner PortVal":"Impact: " + str(impact)}, inplace=True)
        chart_dfs.append(strategy_portval)

        impact += 0.002

    portval_df = pd.concat(chart_dfs, axis=1)

    portval_graph = portval_df.plot(title="Exp. 2: Strategy Learner Changing Impact", fontsize=12, grid=True)
    portval_graph.set_xlabel("Date")
    portval_graph.set_ylabel("Normalized Portfolio Value ($)")

    plt.savefig("Figure_2.png")

    print("------------------------------------------")
