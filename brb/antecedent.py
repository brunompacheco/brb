from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

from interval import interval, inf

from .attr_input import AttributeInput, is_numeric

class Antecedent(ABC):
    """An antecedent attribute for a BRB system.

    Args:
        name: Name of the antecedent.
    """
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def _accepted_dtypes(self):
        """Accepted data types. Return must be acceptable by `isinstance`.
        """

    def match(self, X: AttributeInput, A_i_k: Any) -> float:
        """Computes match between an input and a rule referential value.

        Args:
            X: Input that contains a referential value for this antecedent.
            A_i_k: Referential value for this antecedent "of the k-th rule".

        Returns:
            match_degree: Measures how much the input matches the referential
            value provided, ranges from 0 to 1.
        """
        assert self.name in X.attr_input.keys(), (
            "Input `X` has no values for antecedent {}".format(self.name))

        assert self.is_ref_value(A_i_k)

        match = self._match(X[self.name], A_i_k)

        assert isinstance(match, float)

        return match

    @abstractmethod
    def _match(self, X_i: Any, A_i_k: Any) -> float:
        """Computes match between an input and a rule referential value.

        Args:
            X_i: Input referential value for this antecedent.
            A_i_k: Referential value for this antecedent "of the k-th rule".

        Returns:
            match_degree: Measures how much the input matches the referential
            value provided, ranges from 0 to 1.
        """

    def is_ref_value(self, A_i_k) -> bool:
        """Returns True if `A_i_k` is a valid referential value.
        """
        if isinstance(A_i_k, dict):
            return all(
                [isinstance(k, self._accepted_dtypes) for k in A_i_k.keys()]
            )
        else:
            return isinstance(A_i_k, self._accepted_dtypes)

class ContinuousAntecedent(Antecedent):
    """Implementation of an antecedent over a continuous numerical space.

    Attributes:
        name: Name of the antecedent.
    """
    _accepted_dtypes = (int, float, interval)

    def _match(self, X_i, A_i_k):
        match = 0.0

        if isinstance(A_i_k, dict):
            if isinstance(X_i, dict):
                # In case both are uncertain, the match is computed
                # individually and multiplied by the rule certainty level.
                for ref_value in A_i_k.keys():
                    match += self._match(X_i, ref_value) * A_i_k[ref_value]
            else:
                # the approach for calculating the match for an uncertain rule
                # is just like the approach for calculating the match for an
                # uncertain input.
                match = self._match(A_i_k, X_i)
        elif is_numeric(X_i):
            if is_numeric(A_i_k):
                match = float(X_i == A_i_k)
            elif isinstance(A_i_k, interval):
                match = float(X_i in A_i_k)
        elif isinstance(X_i, interval):
            X_i_length = X_i[0][1] - X_i[0][0]
            # if X_i_length == inf:
                # warn((
                    # 'The use of unbounded intervals as input ({}) is not'
                    # 'advised, resulting match might not follow expectations.'
                # ).format(X_i))

            if is_numeric(A_i_k):
                # In this case, if the input covers the referential value, we
                # consider it a match. We do so in a binary manner because it
                # would be impossible to quantify how much of the input is
                # covered by the referential value, as the latter has no
                # measure.
                match = float(A_i_k in X_i)
            elif isinstance(A_i_k, interval):
                # For this scenario, we quantify the match as the amount of the
                # input that is contained in the referential value.
                intrsc = A_i_k & X_i
                try:
                    intrsc_length = intrsc[0][1] - intrsc[0][0]

                    if X_i_length == inf:
                        if X_i == A_i_k:
                            match = 1.0
                        elif intrsc_length == 0:
                            match = 0.0
                        else:
                            # As no proper way of quantifying the match of
                            # infinite intervals was found, we assume that if
                            # they are not equal but have a non-empty infinite
                            # intersection, it is a 0.5 match.
                            match = 0.5
                    else:
                        match = float(intrsc_length / X_i_length)
                except IndexError:  # intersection is empty
                    match = 0.0
        elif isinstance(X_i, set):
            if is_numeric(A_i_k):
                # Same as the case for interval input and numeric reference.

                match = float(A_i_k in X_i) / len(X_i)
            elif isinstance(A_i_k, interval):
                # Problems might occur due to the nature of the intervals,
                # e.g., if X_i is {1,2} and A_i is [2,3], this would result in a
                # 0.50 match, even though the intervals share only their upper
                # boundary.
                warn((
                    'comparison between integer interval input `{}` and '
                    'continuous interval `{}` is not advised, results might '
                    'not match the expectations.'
                ).format(X_i, A_i_k))

                intrsc_length = sum([
                    X_i_element in A_i_k for X_i_element in X_i
                ])
                X_i_length = len(X_i)

                match = float(intrsc_length / X_i_length)
        elif isinstance(X_i, dict):
            if is_numeric(A_i_k):
                match = float(X_i[A_i_k])
            elif isinstance(A_i_k, interval):
                matching_certainties = [X_i[key] for key in X_i.keys()
                                        if key in A_i_k]

                match = float(sum(matching_certainties))

        return match

class DiscreteAntecedent(Antecedent):
    """Implementation of an antecedent over a discrete numerical space.

    Attributes:
        name: Name of the antecedent.
    """
    _accepted_dtypes = (int, float, set)

    def _match(self, X_i, A_i_k):
        match = 0.0

        if isinstance(A_i_k, dict):
            if isinstance(X_i, dict):
                # In case both are uncertain, the match is computed
                # individually and multiplied by the rule certainty level.
                for ref_value in A_i_k.keys():
                    match += self._match(X_i, ref_value) * A_i_k[ref_value]
            else:
                # the approach for calculating the match for an uncertain rule
                # is just like the approach for calculating the match for an
                # uncertain input.
                match = self._match(A_i_k, X_i)
        elif is_numeric(X_i):
            if is_numeric(A_i_k):
                match = float(X_i == A_i_k)
            elif isinstance(A_i_k, set):
                match = float(X_i in A_i_k)
        elif isinstance(X_i, interval):
            X_i_length = X_i[0][1] - X_i[0][0]
            if X_i_length == inf:
                warn((
                    'The use of unbounded intervals as input ({}) is not'
                    'advised, resulting match might not follow expectations.'
                ).format(X_i))

            if is_numeric(A_i_k):
                # In this case, if the input covers the referential value, we
                # consider it a match. We do so in a binary manner because it
                # would be impossible to quantify how much of the input is
                # covered by the referential value, as the latter has no
                # measure.
                match = float(A_i_k in X_i)
            elif isinstance(A_i_k, set):
                warn((
                    'The referential value ({}) will be converted to a '
                    'continuous interval to make the comparison with the input '
                    '({}). This is not advised as may result in unexpected '
                    'behavior. Please use an integer interval instead or '
                    'convert the rule\'s referential value to continuous'
                ).format(A_i_k, X_i))
                A_i_k_continuous = interval[min(A_i_k), max(A_i_k)]

                # For this scenario, we quantify the match as the amount of the
                # input that is contained in the referential value.
                intrsc = A_i_k_continuous & X_i
                try:
                    intrsc_length = intrsc[0][1] - intrsc[0][0]

                    if X_i_length == inf:
                        if X_i == A_i_k_continuous:
                            match = 1.0
                        elif intrsc_length == 0:
                            match = 0.0
                        else:
                            # As no proper way of quantifying the match of
                            # infinite intervals was found, we assume that if
                            # they are not equal but have a non-empty infinite
                            # intersection, it is a 0.5 match.
                            match = 0.5
                    else:
                        match = float(intrsc_length / X_i_length)
                except IndexError:  # intersection is empty
                    match = 0.0
        elif isinstance(X_i, set):
            if is_numeric(A_i_k):
                # Same as the case for interval input and numeric reference.

                match = float(A_i_k in X_i) / len(X_i)
            elif isinstance(A_i_k, set):
                intrsc_length = len(X_i & A_i_k)
                X_i_length = len(X_i)

                match = float(intrsc_length / X_i_length)
        elif isinstance(X_i, dict):
            if is_numeric(A_i_k):
                match = float(X_i[A_i_k])
            elif isinstance(A_i_k, set):
                matching_certainties = [X_i[key] for key in X_i.keys()
                                        if key in A_i_k]

                match = float(sum(matching_certainties))

        return match

class CategoricalAntecedent(Antecedent):
    """Implementation of an antecedent over a discrete numerical space.

    Attributes:
        name: Name of the antecedent.
    """
    _accepted_dtypes = (int, float, set)

    def _match(self, X_i, A_i_k):
        match = 0.0

        if isinstance(A_i_k, dict):
            if isinstance(X_i, dict):
                # In case both are uncertain, the match is computed
                # individually and multiplied by the rule certainty level.
                for ref_value in A_i_k.keys():
                    match += self._match(X_i, ref_value) * A_i_k[ref_value]
            else:
                # the approach for calculating the match for an uncertain rule
                # is just like the approach for calculating the match for an
                # uncertain input.
                match = self._match(A_i_k, X_i)
        elif isinstance(X_i, str):
            if isinstance(A_i_k, str):
                match = float(X_i == A_i_k)

        return match
