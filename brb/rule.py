from typing import List, Dict, Any, Union, Callable
from warnings import warn

import numpy as np

from interval import interval

from .attr_input import AttributeInput, is_numeric

class Rule():
    """A rule definition in a BRB system.

    It translates expert knowledge into a mapping between the antecedents and
    the consequents. We assume that it is defined as a pure AND rules, that is,
    the only logical relation between the input attributes is the AND function.

    Attributes:
        A_values: A^k. Dictionary that matches reference values for each
        antecedent attribute that activates the rule.
        beta: \bar{\beta}. Expected belief degrees of consequents if rule is
        delta: \delta_k. Relative weights of antecedent attributes. If not
        provided, 1 will be set for all attributes.
        theta: \theta_k. Rule weight.
        matching_degree: \phi. Defines how to calculate the matching degree for
        the rule. If `Callable`, must be a function that takes `delta`,
        and `alphas_i` (dictionary that maps antecedents to their matching
        degree given input) as input. If string, must be either 'geometric'
        (default) or 'arithmetic', which apply the respective weighted means.
    """

    def __init__(
            self,
            A_values: Dict[str, Any],
            beta: List[float],
            delta: Dict[str, float] = None,
            theta: float = 1,
            matching_degree: Union[str, Callable] = 'arithmetic'
        ):
        self.A_values = A_values

        if delta is None:
            self.delta = {attr: 1 for attr in A_values.keys()}
        else:
            # there must exist a weight for all antecedent attributes that
            # activate the rule
            for U_i in A_values.keys():
                assert U_i in delta.keys()
            self.delta = delta

        self.theta = theta
        self.beta = beta

        self.matching_degree = matching_degree

    def get_antecedent_matching(self, U_i, X) -> float:
        """Quantifies matching of an input to the rules' referential value.

        Args:
            U_i: Antecedent to compare.
            X_i: Input to match to the rule.

        Returns:
            match: Between 0-1, quantifies how much `X_i` matches the
            referential value `A_i`.
        """
        X_i = X.attr_input[U_i]
        A_i = self.A_values[U_i]

        _X_i = X[U_i]
        _A_i = X.prep_referential_value(A_i)

        return self._get_antecedent_matching(_X_i, _A_i, X_i, A_i)

    @staticmethod
    def _get_antecedent_matching(_X_i, _A_i, X_i=None, A_i=None):
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
                intrsc_length = intrsc[0][1] - intrsc[0][0]

                _X_i_length = _X_i[0][1] - _X_i[0][0]

                match = float(intrsc_length / _X_i_length)
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

        return match

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (6a).
        """
        self._assert_input(X)

        alphas_i = {
            U_i: self.get_antecedent_matching(U_i, X)
            for U_i in self.A_values.keys()
        }

        if self.matching_degree == 'geometric':
            return self._geometric_matching_degree(self.delta, alphas_i)
        elif self.matching_degree == 'arithmetic':
            return self._arithmetic_matching_degree(self.delta, alphas_i)
        elif callable(self.matching_degree):
            return self.matching_degree(self.delta, alphas_i)

    @staticmethod
    def _arithmetic_matching_degree(
            delta: Dict[str, float],
            alphas_i: Dict[str, float]
        ) -> float:
        norm_delta = {attr: d / sum(delta.values()) for attr, d
                      in delta.items()}
        weighted_alpha = [
            alpha_i * norm_delta[U_i] for U_i, alpha_i in alphas_i.items()
        ]

        return np.sum(weighted_alpha)

    @staticmethod
    def _geometric_matching_degree(
            delta: Dict[str, float],
            alphas_i: Dict[str, float]
        ) -> float:
        norm_delta = {attr: d / max(delta.values()) for attr, d
                      in delta.items()}
        weighted_alpha = [
            alpha_i ** norm_delta[U_i] for U_i, alpha_i in alphas_i.items()
        ]

        return np.prod(weighted_alpha)

    def get_belief_degrees_complete(self, X: AttributeInput) -> Dict[Any, Any]:
        """Returns belief degrees transformed based on input completeness

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (8).
        """
        self._assert_input(X)

        rule_input_completeness = X.get_completeness(self.A_values.keys())

        norm_beta = [belief * rule_input_completeness for belief in self.beta]

        return norm_beta

    def _assert_input(self, X: AttributeInput):
        """Checks if `X` is proper.

        Guarantees that all the necessary attributes are present in X.
        """
        rule_attributes = set(self.A_values.keys())
        input_attributes = set(X.attr_input.keys())
        assert rule_attributes.intersection(input_attributes) == rule_attributes

    def __str__(self):
        A_values_str = ["({}:{})".format(U_i, A_i)
                        for U_i, A_i
                        in self.A_values.items()]

        str_out = r' /\ '.join(A_values_str)

        # TODO: add consequents labels
        str_out += ' => ' + str(self.beta)

        return str_out
