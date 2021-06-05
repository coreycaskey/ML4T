"""
    Assess a Roulette Wheel betting strategy (using an American wheel)
"""

import matplotlib.pyplot as plt
import numpy as np

#
def test_code():
    win_prob = 0.474 # 18/38 - red, 18/38 - black
    np.random.seed(1)

    # Experiment 1
    make_plot(win_prob, figure_num=1)
    make_plot(win_prob, figure_num=2)
    make_plot(win_prob, figure_num=3)

    # Experiment 2
    make_plot(win_prob, figure_num=4)
    make_plot(win_prob, figure_num=5)

#
def make_plot(win_prob, figure_num):
    """
        General functionality for all plots
    """

    plt.figure(figure_num)
    plt.axis([0, 300, -256, 100])

    plt.xlabel('Number of Bets')
    plt.ylabel('Total Earnings ($)')

    if figure_num == 1:
        figure_one_test(win_prob)
    elif figure_num == 2:
        figure_two_test(win_prob)
    elif figure_num == 3:
        figure_three_test(win_prob)
    elif figure_num == 4:
        figure_four_test(win_prob)
    elif figure_num == 5:
        figure_five_test(win_prob)

    plt.savefig('Figure_{}.png'.format(figure_num))

#
def figure_one_test(win_prob):
    """
        Run the simple simulator 10 times (with max 1000 spins) and plot each line
    """

    for _ in range(10):
        plt.plot(run_simple_bet_sim(win_prob, max_spin_count=1000, winnings_limit=80))

    plt.title('Figure 1: 10 Simulations')

#
def figure_two_test(win_prob):
    """
        Run the simple simulator 1000 times (with max 1000 spins) and plot the mean and
        standard deviation lines
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = get_simple_simulations_matrix(win_prob, max_spin_count=1000, num_simulations=1000)

    mean_winnings_for_spins = np.mean(simulations_matrix, axis=0) # calculate for each spin
    std_winnings_for_spins = np.std(simulations_matrix, axis=0)

    std_plus = mean_winnings_for_spins + std_winnings_for_spins
    std_minus = mean_winnings_for_spins - std_winnings_for_spins

    add_mean_legend(mean_winnings_for_spins, std_plus, std_minus, 'lower right')

    plt.title('Figure 2: 1000 Simulations (Mean & Std)')

#
def figure_three_test(win_prob):
    """
        Run the simple simulator 1000 times (with max 1000 spins) and plot the median and
        standard deviation lines
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = get_simple_simulations_matrix(win_prob, max_spin_count=1000, num_simulations=1000)

    median_winnings_for_spins = np.median(simulations_matrix, axis=0) # calculate for each spin
    std_winnings_for_spins = np.std(simulations_matrix, axis=0)

    std_plus = median_winnings_for_spins + std_winnings_for_spins
    std_minus = median_winnings_for_spins - std_winnings_for_spins

    add_median_legend(median_winnings_for_spins, std_plus, std_minus, 'lower right')

    plt.title('Figure 3: 1000 Simulations (Median & Std)')

#
def figure_four_test(win_prob):
    """
        Run the realistic simulator 1000 times (with max 1000 spins) and plot the mean and
        standard deviation lines
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = get_real_simulations_matrix(win_prob, max_spin_count=1000, num_simulations=1000)

    mean_winnings_for_spins = np.mean(simulations_matrix, axis=0) # calculate for each spin
    std_winnings_for_spins = np.std(simulations_matrix, axis=0)

    std_plus = mean_winnings_for_spins + std_winnings_for_spins
    std_minus = mean_winnings_for_spins - std_winnings_for_spins

    add_mean_legend(mean_winnings_for_spins, std_plus, std_minus, 'lower left')

    plt.title('Figure 4: 1000 Realistic Simulations (Mean & Std)')

#
def figure_five_test(win_prob):
    """
        Run the realistic simulator 1000 times (with max 1000 spins) and plot the median and
        standard deviation lines
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = get_real_simulations_matrix(win_prob, max_spin_count=1000, num_simulations=1000)

    median_winnings_for_spins = np.median(simulations_matrix, axis=0) # calculate for each spin
    std_winnings_for_spins = np.std(simulations_matrix, axis=0)

    std_plus = median_winnings_for_spins + std_winnings_for_spins
    std_minus = median_winnings_for_spins - std_winnings_for_spins

    add_median_legend(median_winnings_for_spins, std_plus, std_minus, 'lower right')

    plt.title('Figure 5: 1000 Realistic Simulations (Median & Std)')

#
def get_simple_simulations_matrix(win_prob, max_spin_count, num_simulations):
    """
        Generates a matrix of winnings per spin for some number of simulations
        (using the simple bet simulator)
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = np.zeros((num_simulations, max_spin_count + 1))

    for sim_count in range(num_simulations):
        simulations_matrix[sim_count] = run_simple_bet_sim(win_prob, max_spin_count=max_spin_count, winnings_limit=80)

    return simulations_matrix

#
def get_real_simulations_matrix(win_prob, max_spin_count, num_simulations):
    """
        Generates a matrix of winnings per spin for some number of simulations
        (using the realistic bet simulator)
    """

    # matrix of winnings where row = simulation number and col = spin number
    simulations_matrix = np.zeros((num_simulations, max_spin_count + 1))

    for sim_count in range(num_simulations):
        simulations_matrix[sim_count] = run_real_bet_sim(win_prob, max_spin_count=max_spin_count, winnings_limit=80, initial_money=256)

    return simulations_matrix

#
def run_simple_bet_sim(win_prob, max_spin_count, winnings_limit):
    """
        We employ our betting strategy until we exceed the max_spin_count
        or reach the winnings_limit (with the assumption that we have an
        infinite amount of money to spend)
    """

    winnings_tracker = np.zeros(max_spin_count + 1) # track winnings after each spin, start at 0th spin
    total_winnings = 0
    bet_amount = 1
    spin_count = 1

    while spin_count <= max_spin_count and total_winnings < winnings_limit:
        total_winnings, bet_amount = use_bet_strategy(total_winnings, bet_amount, win_prob)

        winnings_tracker[spin_count] = total_winnings
        spin_count += 1

    # if we reach or exceed our winnings limit, cascade it for the remaining spins
    if total_winnings >= winnings_limit:
        winnings_tracker[spin_count:] = winnings_limit

    return winnings_tracker

#
def run_real_bet_sim(win_prob, max_spin_count, winnings_limit, initial_money):
    """
        We employ our betting strategy until we exceed the max_spin_count,
        reach the winnings_limit, or lose all of our money
    """

    winnings_tracker = np.zeros(max_spin_count + 1) # track winnings after each spin, start at 0th spin
    total_winnings = 0
    bet_amount = 1
    spin_count = 1

    while spin_count <= max_spin_count and total_winnings < winnings_limit and total_winnings > -initial_money:
        if bet_amount > total_winnings + initial_money:
            bet_amount = total_winnings + initial_money

        total_winnings, bet_amount = use_bet_strategy(total_winnings, bet_amount, win_prob)

        winnings_tracker[spin_count] = total_winnings
        spin_count += 1

    # if we reach or exceed our winnings limit, cascade it for the remaining spins
    if total_winnings >= winnings_limit:
        winnings_tracker[spin_count:] = winnings_limit

    # if we lose all our money, cascade it for the remaining spins
    if total_winnings <= -initial_money:
        winnings_tracker[spin_count:] = -initial_money

    return winnings_tracker

#
def use_bet_strategy(total_winnings, bet_amount, win_prob):
    """
        Strategy relies on betting $1 every time we win and doubling
        our bet everytime we lose
    """

    return (total_winnings + bet_amount, 1) \
        if get_spin_result(win_prob) \
        else (total_winnings - bet_amount, bet_amount * 2)

#
def get_spin_result(win_prob):
    return True if np.random.random() <= win_prob else False

#
def add_mean_legend(mean_line, std_plus_line, std_minus_line, position):
    """
        Shared functionality for plots that require a mean legend
    """

    plt.plot(mean_line, label='Mean')
    plt.plot(std_plus_line, label='Mean + Std')
    plt.plot(std_minus_line, label='Mean - Std')

    plt.legend(loc=position, shadow=True, fontsize='medium')

#
def add_median_legend(median_line, std_plus_line, std_minus_line, position):
    """
        Shared functionality for plots that require a median legend
    """

    plt.plot(median_line, label='Median')
    plt.plot(std_plus_line, label='Median + Std')
    plt.plot(std_minus_line, label='Median - Std')

    plt.legend(loc=position, shadow=True, fontsize='medium')

#
if __name__ == '__main__':
    test_code()
