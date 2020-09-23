import numpy as np
from typing import List, Dict, Any, Union, Callable
from warnings import warn
import math

import numpy as np

a = 10
b = 13

mod = math.floor(b / a)
modu = b % a

print(mod, modu)

from interval import interval, inf


def is_numeric(a) -> bool:  # pylint: disable=missing-function-docstring
    try:
        float(a)
        return True
    except:
        return False

lst = [0,1,2,3,4]
print(len(lst))

X_i = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
A_i = '>100'
_A_i = interval([10000, inf])
_X_i = interval([20, inf])
match = 0.0

intrsc = _A_i & _X_i
_X_i_length = _X_i[0][1] - _X_i[0][0]
try:
    intrsc_length = intrsc[0][1] - intrsc[0][0]

    if _X_i_length == inf:
        # As no proper way of quantifying the match of infinite
        # intervals was found, we assume that if they are not
        # equal but have a non-empty infinite intersection, it
        # is a 0.5 match.
        if _X_i == _A_i:
            match = 1.0
        elif intrsc_length == 0:
            match = 0.0
        else:
            match = 0.5
    else:
        match = float(intrsc_length / _X_i_length)
except IndexError:  # intersection is empty
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
        print('both are intervals')
        intrsc = _A_i & _X_i
        try:
            end = intrsc[0][1]
            start = intrsc[0][0]
            intrsc_length = end - start

            _X_i_length = _X_i[0][1] - _X_i[0][0]

            if intrsc_length == inf and _X_i_length == inf:
                if _X_i[0][1] == inf and _A_i[0][0] <= _X_i[0][0]:
                    match = 1.0
                else:
                    match = 0.0
            else:
                match = float(intrsc_length / _X_i_length)
        except IndexError:  # intersection is empty
            match = 0.0

elif isinstance(_X_i, set):
    if is_numeric(_A_i):
        # Same as the case for interval input and numeric reference.

        match = float(_A_i in _X_i) / len(_X_i)
    elif isinstance(_A_i, set):
        intrsc_length = len(_X_i & _A_i)
        _X_i_length = len(_X_i)

        match = float(intrsc_length / _X_i_length)
    elif isinstance(_A_i, interval):
        warn((
                 'comparison between integer interval input `{}` and '
                 'continuous interval `{}` not supported.'
             ).format(X_i, A_i))
        _X_i = list(_X_i)
        if _A_i[0][0] <= _X_i[0] and _A_i[0][1] >= _X_i[-1] :
            match = 1.0

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
if np.isnan(match):
    print('nan just created')
    print('why does he stop here')

print(match)