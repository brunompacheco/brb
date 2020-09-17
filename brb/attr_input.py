"""Implementation of input object and helper function for BRB model input.

This module provides tools that are able to convert data in string or pythonic
form to data types suitable for the model to process. These formats include
support for uncertainty and incompleteness, and also for intervals.
"""
from ast import literal_eval
from typing import Dict, Any, List, Union

from interval import interval, inf


def str2interval(value: str) -> Union[interval, set]:
    """Converts a string to an integer or real-valued interval.

    An interval must match one of the three following formats: "s:e", ">s" or
    "<e", in which 's' and 'e' are the interval's start and end boundaries. If
    either of these numbers is a `float`, then the returning interval is
    real-valued. If the open formats is used, a real-valued interval is returned
    due to `int` data type lack of support for `inf`. 

    Returns:
        value_interval: Integer intervals are represented as sets, while
        real-valued intervals are represented through pyinterval's interval
        object.

    Raises:
        ValueError: In case the provided string is not formatted as one of the
        three formats described.
    """
    _value = value.strip()

    INTERVAL_SEP = ':'
    BT_SEP = '>'
    ST_SEP = '<'
    if INTERVAL_SEP in _value:
        start, end = _value.split(INTERVAL_SEP)
    elif BT_SEP in _value:
        start = _value.split(BT_SEP)[-1]
        end = inf
    elif ST_SEP in _value:
        start = 0
        end = _value.split(ST_SEP)[-1]
    else:
        raise ValueError('`{}` is not a proper interval'.format(value))

    try:
        if start != -inf:
            start = int(start)
        if end != inf:
            end = int(end)
    except ValueError:
        start = float(start)
        end = float(end)

    if isinstance(start, int) and isinstance(end, int):
        value_interval = set(range(start, end + 1))
    else:
        value_interval = interval[start, end]

    return value_interval

def is_numeric(a) -> bool:  # pylint: disable=missing-function-docstring
    try:
        float(a)
        return True
    except:
        return False

class AttributeInput():
    """An input to the BRB system.

    Consists of a set of antecedent attribute values and degrees of belief.

    Attributes:
        attr_input: X. Relates antecedent attributes with values and belief
        degrees. Must follow same order of reference values as in the model.
    """

    def __init__(self, attr_input: Dict[str, Dict[Any, float]]):
        self.attr_input = attr_input

    def __getitem__(self, key):
        """Prepares the object's value before returning.
        """
        return self.prep_referential_value(self.attr_input[key])

    @staticmethod
    def prep_referential_value(X_i):
        """Coerces pythonic string input to acceptable data type.

        Returns:
            _X_i: Either a numerical value (`int` or `float`), an interval
            (`set` or `interval`, see `str2interval` function), a string
            (categorical value), or a dictionary, which is understood as an
            uncertain distribution over categorical or numerical values.
        """
        try:
            # TODO: rewrite code based on `literal_eval` application on the
            # input
            _X_i = literal_eval(X_i)
        except (ValueError, SyntaxError):
            _X_i = X_i

        # TODO: add dictionary values check

        if isinstance(_X_i, str):
            # strips whitespaces from the inputs
            _X_i = _X_i.strip()

            # converts to numeric if so
            if is_numeric(_X_i):
                try:
                    _X_i = int(_X_i)
                except ValueError:
                    _X_i = float(_X_i)
            else:
                # if not numeric, try interval
                try:
                    _X_i = str2interval(_X_i)
                except:
                    # if not interval, then understand it as string/categorical
                    return X_i

        return _X_i

    def get_completeness(self, A: List[Any]) -> float:
        """Returns input completeness given set of antecedents.

        The completeness is a value from 0 to 1 that quantifies the completeness
        of the uncertain distribution of the input. It is computed only over `A`
        as it is understood that completeness is only analysed over the
        antecedents defined by a rule (that is the intended caller).

        Args:
            A: List of antecedents that the completeness should be computed
            over.

        Returns:
            completeness: Measures how complete (if there is 1.0 *sum* of
            uncertainty) the input is, 1.0 being totally complete and 0.0 being
            totally incomplete.
        """
        sum_completeness = 0.0
        for A_i in A:
            try:
                if isinstance(self[A_i], dict):
                    sum_completeness += sum(self[A_i].values())
                else:
                    sum_completeness += 1.0
            except KeyError:
                sum_completeness += 0.0

        return sum_completeness / len(A)
