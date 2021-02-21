"""
  Optimize a Portfolio - Grading Script

  Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python3 grade_optimization.py`
"""

from collections import namedtuple
from grading.grading import GradeResult, IncorrectOutput, grader, time_limit

import datetime
import numpy as np
import os
import pytest
import sys
import traceback as tb

main_code = 'optimization'

#
def str_to_date(date_str):
  year, month, day = map(int, date_str.split('-'))
  return datetime.datetime(year, month, day)

# test cases
OptimizationTestCase = namedtuple('OptimizationTestCase', ['inputs', 'outputs', 'description'])

optimization_test_cases = [
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2010-01-01'),
      end_date=str_to_date('2010-12-31'),
      symbols=['GOOG', 'AAPL', 'GLD', 'XOM'],
    ),
    outputs=dict(allocs=[0.0, 0.4, 0.6, 0.0]),
    description='Wiki Example 1',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2004-01-01'),
      end_date=str_to_date('2006-01-01'),
      symbols=['AXP', 'HPQ', 'IBM', 'HNZ'],
    ),
    outputs=dict(allocs=[0.78, 0.22, 0.0, 0.0]),
    description='Wiki Example 2',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2004-12-01'),
      end_date=str_to_date('2006-05-31'),
      symbols=['YHOO', 'XOM', 'GLD', 'HNZ'],
    ),
    outputs=dict(allocs=[0.0, 0.07, 0.59, 0.34]),
    description='Wiki Example 3',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2005-12-01'),
      end_date=str_to_date('2006-05-31'),
      symbols=['YHOO', 'HPQ', 'GLD', 'HNZ'],
    ),
    outputs=dict(allocs=[0.0, 0.1, 0.25, 0.65]),
    description='Wiki Example 4',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2005-12-01'),
      end_date=str_to_date('2007-05-31'),
      symbols=['MSFT', 'HPQ', 'GLD', 'HNZ'],
    ),
    outputs=dict(allocs=[0.0, 0.27, 0.11, 0.62]),
    description='MSFT vs HPQ',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2006-05-31'),
      end_date=str_to_date('2007-05-31'),
      symbols=['MSFT', 'AAPL', 'GLD', 'HNZ'],
    ),
    outputs=dict(allocs=[0.42, 0.32, 0.0, 0.26]),
    description='MSFT vs AAPL',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2011-01-01'),
      end_date=str_to_date('2011-12-31'),
      symbols=['AAPL', 'GLD', 'GOOG', 'XOM'],
    ),
    outputs=dict(allocs=[0.46, 0.37, 0.0, 0.17]),
    description='Wiki Example 1 in 2011',
  ),
  OptimizationTestCase(
    inputs=dict(
      start_date=str_to_date('2010-01-01'),
      end_date=str_to_date('2010-12-31'),
      symbols=['AXP', 'HPQ', 'IBM', 'HNZ'],
    ),
    outputs=dict(allocs=[0.0, 0.0, 0.0, 1.0]),
    description='Year of the HNZ',
  )
]

abs_margins = dict(sum_to_one=0.02, alloc_range=0.02, alloc_match=0.1)  # absolute margin of error for each component
points_per_component = dict(sum_to_one=2.0, alloc_range=2.0, alloc_match=4.0) # points for each component, for partial credit
points_per_test_case = sum(points_per_component.values())
seconds_per_test_case = 10  # execution time limit

# grading parameters (picked up by module-level grading fixtures)
max_points = float(len(optimization_test_cases) * points_per_test_case)
html_pre_block = True # surround comments with HTML <pre> tag

# test function(s)
@pytest.mark.parametrize('inputs,outputs,description', optimization_test_cases)
def test_optimization(inputs, outputs, description, grader):
  """
    Test that optimize_portfolio() returns correct allocations

    Requires test inputs, expected outputs, description, and a grader fixture
  """

  points_earned = 0.0  # initialize points for this test case

  try:
    # try to import student code (only once)
    if not main_code in globals():
      import importlib

      mod = importlib.import_module(main_code)
      globals()[main_code] = mod

    # unpack test case
    start_date = inputs['start_date']
    end_date = inputs['end_date']
    symbols = inputs['symbols'] # e.g. ['GOOG', 'AAPL', 'GLD', 'XOM']

    # run student code with time limit (in seconds, per test case)
    with time_limit(seconds_per_test_case):
      student_allocs, student_cr, student_adr, student_sddr, student_sr = \
        optimization.optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=False)

      student_allocs = np.float32(student_allocs) # use a NumPy array, for easier computation

    # verify against expected outputs and assign points
    incorrect = False
    msgs = []
    correct_allocs = outputs['allocs']

    # check sum_to_one: allocations sum to 1.0 +/- margin
    sum_allocs = np.sum(student_allocs)

    if abs(sum_allocs - 1.0) > abs_margins['sum_to_one']:
      incorrect = True
      msgs.append('    sum of allocations: {} (expected: 1.0)'.format(sum_allocs))
      student_allocs = student_allocs / sum_allocs  # normalize allocations, if they don't sum to 1.0

    else:
      points_earned += points_per_component['sum_to_one']

    # check alloc_range: each allocation is within [0.0, 1.0] +/- margin
    points_per_alloc_range = points_per_component['alloc_range'] / len(correct_allocs)

    # check alloc_match: each allocation matches expected value +/- margin
    points_per_alloc_match = points_per_component['alloc_match'] / len(correct_allocs)

    for symbol, alloc, correct_alloc in zip(symbols, student_allocs, correct_allocs):
      if alloc < -abs_margins['alloc_range'] or alloc > (1.0 + abs_margins['alloc_range']):
        incorrect = True
        msgs.append('    {} - allocation out of range: {} (expected: [0.0, 1.0])'.format(symbol, alloc))

      else:
        points_earned += points_per_alloc_range

        if abs(alloc - correct_alloc) > abs_margins['alloc_match']:
          incorrect = True
          msgs.append('    {} - incorrect allocation: {} (expected: {})'.format(symbol, alloc, correct_alloc))

        else:
          points_earned += points_per_alloc_match

    if incorrect:
      inputs_str = '    start_date: {}\n    end_date: {}\n    symbols: {}\n'.format(start_date, end_date, symbols)
      points_earned = 0

      raise IncorrectOutput('Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}'.format(inputs_str, '\n'.join(msgs)))

  except Exception as e:
    # test result: failed
    msg = 'Test case description: {}\n'.format(description)

    # generate a filtered stacktrace, only showing erroneous lines in student file(s)
    tb_list = tb.extract_tb(sys.exc_info()[2])

    for i in range(len(tb_list)):
      row = tb_list[i]
      tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3]) # show only filename instead of long absolute path

    tb_list = [ row for row in tb_list if row[0] == 'optimization.py' ]

    if tb_list:
      msg += 'Traceback:\n'
      msg += ''.join(tb.format_list(tb_list)) # contains newlines

    msg += '{}: {}'.format(e.__class__.__name__, str(e))

    # report failure result to grader, with stacktrace
    grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
    raise

  else:
    # test result: passed (no exceptions)
    grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

#
if __name__ == '__main__':
  # the '-s' flag disables capturing, showing stdcalls for print statements, logging calls, etc.
  # __file__ points to this file path

  pytest.main(['-s', __file__])
