
#import numpy as np
#from typing import List, Dict, Any, Union, Callable
#from warnings import warn
#import math
#from brb.rule import Rule
#import numpy as np


counter = 0
while counter < 1000000:
    print(counter)
    counter += 1

'''
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

_X_i1 = str2interval('3:7')
_X_i2 = str2interval('<7')
_X_i3 = str2interval('<5.0')
_X_i4 = str2interval('3.0:7')

_A_iset = str2interval('3:5')
_A_iinterv = str2interval('>3.0')

def is_numeric(a) -> bool:  # pylint: disable=missing-function-docstring
    try:
        float(a)
        return True
    except:
        return False

def _get_antecedent_matching(_X_i, _A_i, X_i=None, A_i=None) -> float:
    """Calculates match level between the two inputs.

    Args:
        _X_i: Input referential value already in a data type that the model
        handles (see `AttributeInput.prep_referential_value`).
        _A_i: Rule antecedent referential value already in a data type that
        the model handles (see `AttributeInput.prep_referential_value`).
        X_i: If provided, will be understood as the string representation of
        `_X_i`.
        A_i: If provided, will be understood as the string representation of
        `_A_i`.
    Returns:
        match: Bounded between 0.0 and 1.0, represents the match between the
        two inputs.
    """
    if X_i is None:
        X_i = _X_i

    if A_i is None:
        A_i = _A_i

    match = 0.0

    if is_numeric(_X_i):
        if is_numeric(_A_i):
            match = float(_X_i == _A_i)
        elif isinstance(_A_i, interval) or isinstance(_A_i, set):
            match = float(_X_i in _A_i)
    elif isinstance(_X_i, str):
        if isinstance(_A_i, str):
            match = float(_X_i == _A_i)
    elif isinstance(_X_i, interval):
        _X_i_length = _X_i[0][1] - _X_i[0][0]
        if _X_i_length == inf:
            warn((
                     'The use of unbounded intervals as input ({}) is not'
                     'advised, resulting match might not follow expectations.'
                 ).format(X_i))

        if is_numeric(_A_i):
            # In this case, if the input covers the referential value, we
            # consider it a match. We do so in a binary manner because it
            # would be impossible to quantify how much of the input is
            # covered by the referential value, as the latter has no
            # measure.
            match = float(_A_i in _X_i)
        elif isinstance(_A_i, interval):
            # For this scenario, we quantify the match as the amount of the
            # input that is contained in the referential value.
            intrsc = _A_i & _X_i
            try:
                intrsc_length = intrsc[0][1] - intrsc[0][0]

                if _X_i_length == inf:
                    if _X_i == _A_i:
                        match = 1.0
                    elif intrsc_length == 0:
                        match = 0.0
                    else:
                        # As no proper way of quantifying the match of infinite
                        # intervals was found, we assume that if they are not
                        # equal but have a non-empty infinite intersection, it
                        # is a 0.5 match.
                        match = 0.5
                else:
                    match = float(intrsc_length / _X_i_length)
            except IndexError:  # intersection is empty
                match = 0.0
        elif isinstance(_A_i, set):
            warn((
                     'The referential value ({}) will be converted to a '
                     'continuous interval to make the comparison with the input '
                     '({}). This is not advised as may result in unexpected '
                     'behavior. Please use an integer interval instead or '
                     'convert the rule\'s referential value to continuous'
                 ).format(A_i, X_i))
            _A_i_continuous = interval[min(_A_i), max(_A_i)]

            match = Rule._get_antecedent_matching(
                _X_i,
                _A_i_continuous,
                X_i,
                A_i
            )
    elif isinstance(_X_i, set):
        if is_numeric(_A_i):
            # Same as the case for interval input and numeric reference.

            match = float(_A_i in _X_i) / len(_X_i)
        elif isinstance(_A_i, set):
            intrsc_length = len(_X_i & _A_i)
            _X_i_length = len(_X_i)

            match = float(intrsc_length / _X_i_length)
        elif isinstance(_A_i, interval):
            # Problems might occur due to the nature of the intervals,
            # e.g., if X_i is {1,2} and A_i is [2,3], this would result in a
            # 0.50 match, even though the intervals share only their upper
            # boundary.
            warn((
                     'comparison between integer interval input `{}` and '
                     'continuous interval `{}` is not advised, results might '
                     'not match the expectations.'
                 ).format(X_i, A_i))

            intrsc_length = sum([
                _X_i_element in _A_i for _X_i_element in _X_i
            ])
            _X_i_length = len(_X_i)

            match = float(intrsc_length / _X_i_length)
    elif isinstance(_X_i, dict):
        if isinstance(_A_i, str) or is_numeric(_A_i):
            match = float(_X_i[_A_i])
        elif isinstance(_A_i, interval) or isinstance(_A_i, set):
            matching_certainties = [_X_i[key] for key in _X_i.keys()
                                    if key in _A_i]

            match = float(sum(matching_certainties))
        elif isinstance(_A_i, dict):
            raise NotImplementedError('Uncertain rules are not supported')
    else:
        warn('Input {} mismatches the referential value {}'.format(
            X_i, A_i
        ))

    assert isinstance(match, float)

    return match

match1 = _get_antecedent_matching(_X_i1, _A_iset)
match2 = _get_antecedent_matching(_X_i2, _A_iset)
match3 = _get_antecedent_matching(_X_i3, _A_iset)
match4 = _get_antecedent_matching(_X_i4, _A_iset)
match5 = _get_antecedent_matching(_X_i1, _A_iinterv)
match6 = _get_antecedent_matching(_X_i2, _A_iinterv)
match7 = _get_antecedent_matching(_X_i3, _A_iinterv)
match8 = _get_antecedent_matching(_X_i4, _A_iinterv)

print(match1, match2, match3, match4, match5, match6, match7, match8)
print('done')
'''