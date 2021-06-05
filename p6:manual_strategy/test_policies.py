"""
    Evaluates the Manul Strategy and Theoretically Optimal Strategy against our benchmark strategy

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python test_policies.py`
"""

from ManualStrategy import ManualStrategy
from TheoreticallyOptimalStrategy import TheoreticallyOptimalStrategy

#
if __name__ == '__main__':
    ms = ManualStrategy()
    tos = TheoreticallyOptimalStrategy()

    ms.evaluate()
    tos.evaluate()
