"""Models a belief rule and associated operations.
"""
from copy import copy
from typing import List, Dict, Any, Union, Callable
from warnings import warn

import numpy as np

from interval import interval, inf

from .antecedent import Antecedent
from .attr_input import AttributeInput, is_numeric

class Rule():
    """A rule definition in a BRB system.

    It translates expert knowledge into a mapping between the antecedents and
    the consequents. We assume that it is defined as a pure AND rule, that is,
    the only logical relation between the input attributes is the AND function.

    Attributes:
        U: Antecedents that the rules account for.
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
            A_values: Dict[Antecedent, Any],
            beta: List[float],
            delta: Dict[str, float] = None,
            theta: float = 1,
            matching_degree: Union[str, Callable] = 'arithmetic'
        ):
        self.U = A_values.keys()

        self.A_values = A_values
        for U_i in self.U:
            A_i_k = AttributeInput.prep_referential_value(A_values[U_i])

            assert U_i.is_ref_value(A_i_k)

            self.A_values[U_i] = A_i_k

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

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (6a).
        """
        self._assert_input(X)

        alphas_i = {
            U_i: U_i.match(X, self.A_values[U_i])
            for U_i in self.U
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
        """Computes arithmetic average of the antecedents' matching degrees.
        """
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
        """Computes geometric average of the antecedents' matching degrees.
        """
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

    def get_antecedents_names(self):
        names = [U_i.name for U_i in self.U]

        return names

    def _assert_input(self, X: AttributeInput):
        """Checks if `X` is proper.

        Guarantees that all the necessary attributes are present in X.
        """
        rule_attributes = set(self.get_antecedents_names())
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

    def expand_antecedent(self, U_i: str, A_i: list) -> list:
        """Expands itself antecedent into multiple, complete rules.

        In case `U_i` referential values are not provided in the rule
        definition, generates copy of itself covering all the possible values
        this antecedent can take.

        Args:
            U_i: Antecedent name to be used as a base for expansion.
            A_i: All possible referential values for antecedent `U_i`.

        Returns:
            new_rules: List of the new rules generate as copies of `self`
            covering all possibilities for `U_i`.
        """
        # the rule must be empty for the antecedent
        assert U_i not in self.A_values.keys()

        new_rules = list()
        for A_i_j in A_i:
            new_A_values = copy(self.A_values)
            new_A_values[U_i] = A_i_j

            new_delta = copy(self.delta)
            # TODO: implement more robust delta calculation for new antecedent.
            new_delta[U_i] = sum(self.delta.values()) / len(self.delta)

            new_rule = Rule(
                A_values=new_A_values,
                beta=self.beta,
                delta=new_delta,
                theta=self.theta,
                matching_degree=self.matching_degree
            )

            new_rules.append(new_rule)

        return new_rules
