"""
    Experiment 1

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python experiment1.py`
"""

import math
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util import get_data, plot_data
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

#
def compare(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), impact=0.0):
    ms = ManualStrategy()
    sl = StrategyLearner(impact=0.002)

    manual_orders = ms.test_policy(symbol, sd=sd, ed=ed)
    manual_trades = 0

    for date, row in manual_orders.iterrows():
        if row['Shares'] != 0:
            manual_trades += 1

    manual = compute_portvals(manual_orders, start_val=100000, commission=0.0, impact=impact)
    manual_portval = manual.to_frame(name='Manual Strategy PortVal')
    manual_portval /= manual_portval.iloc[0]

    ##########################

    sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)

    trades = sl.test_policy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    trades['Symbol'] = symbol
    trades['Order'] = 'BUY'

    strategy_trades = 0

    for date, row in trades.iterrows():
        if row['Shares'] == -1000:
            trades.at[date, 'Order'] = 'SELL'

        if row['Shares'] != 0:
            strategy_trades += 1

    strategy = compute_portvals(trades, start_val=100000, commission=0.0, impact=impact)

    strategy_portval = strategy.to_frame(name='Strategy Learner PortVal')
    strategy_portval /= strategy_portval.iloc[0]

    portval_df = pd.concat([strategy_portval, manual_portval], axis=1)

    portval_graph = portval_df.plot(title="Exp. 1: Strategy Learner vs. & Manual Strategy (In-Sample)", fontsize=12, grid=True, color=['blue', 'black'])
    portval_graph.set_xlabel("Date")
    portval_graph.set_ylabel("Normalized Portfolio Value ($)")

    plt.savefig("Figure_1.png")

    manual_cr = (manual_portval.iloc[-1].at['Manual Strategy PortVal'] / manual_portval.iloc[0].at['Manual Strategy PortVal']) - 1
    manual_adr = manual_portval.pct_change(1).mean()['Manual Strategy PortVal']
    manual_sddr = manual_portval.pct_change(1).std()['Manual Strategy PortVal']
    manual_sr = math.sqrt(252.0) * (manual_adr / float(manual_sddr))

    strategy_cr = (strategy_portval.iloc[-1].at['Strategy Learner PortVal'] / strategy_portval.iloc[0].at['Strategy Learner PortVal']) - 1
    strategy_adr = strategy_portval.pct_change(1).mean()['Strategy Learner PortVal']
    strategy_sddr = strategy_portval.pct_change(1).std()['Strategy Learner PortVal']
    strategy_sr = math.sqrt(252.0) * (strategy_adr / float(strategy_sddr))

    print("In Sample")

    print("Date Range: {} to {}".format(sd, ed))

    print("Cumulative Return of Strategy: {}".format(strategy_cr))
    print("Cumulative Return of Manual: {}".format(manual_cr))

    print("Standard Deviation of Strategy: {}".format(strategy_sddr))
    print("Standard Deviation of Manual: {}".format(manual_sddr))

    print("Average Daily Return of Strategy: {}".format(strategy_adr))
    print("Average Daily Return of Manual: {}".format(manual_adr))

    print("Sharpe Ratio of Strategy: {}".format(strategy_sr))
    print("Sharpe Ratio of Manual: {}".format(manual_sr))

    print("Number of Trades for Strategy: {}".format(strategy_trades))
    print("Number of Trades for Manual: {}".format(manual_trades))

#
if __name__ == "__main__":
    compare()
