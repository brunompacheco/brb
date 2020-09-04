from ast import literal_eval
from typing import Dict, Any, List

from interval import interval, inf

def str2interval(value: str):
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
        start = -inf
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

def is_numeric(a):  # pylint: disable=missing-function-docstring
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
        return self.prep_referential_value(self.attr_input[key])

    @staticmethod
    def prep_referential_value(X_i):
        """Converts pythonic string input to acceptable data type.
        """
        try:
            # TODO: rewrite code based on `literal_eval` application on the input
            _X_i = literal_eval(X_i)
        except (ValueError, SyntaxError):
            _X_i = X_i

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

    def get_completeness(self, A: List[Any]):
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
