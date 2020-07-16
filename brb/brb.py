"""BRB RIMER implementation.

Based on "Belief Rule-Base Inference Methodology Using the Evidential Reasoning
Approach - RIMER", by _Yang et al._, this module implements three data
structures to construct expert systems based on the Evidential Reasoning
approach.

    Typical usage example:

    >>> from brb.brb import RuleBaseModel, Rule, AttributeInput
    >>> model = RuleBaseModel(U=['Antecedent'], A={'Antecedent': ['good', 'bad']}, D=['good', 'bad'])
    >>> model.add_rule(Rule(
    ...     A_values={'Antecedent':'good'},
    ...     beta=[1,0]
    ... ))
    >>> model.add_rule(Rule(
    ...     A_values={'Antecedent':'bad'},
    ...     beta=[0,1]
    ... ))
    >>> X = AttributeInput({'Antecedent': {'good': 1, 'bad': 0}})
    >>> model.run(X)
    [1.0, 0.0]
    >>> X = AttributeInput({'Antecedent': {'good': 0.3, 'bad': 0.7}})
    >>> model.run(X)
    [0.15517241379310348, 0.8448275862068964]
"""
from typing import List, Dict, Any

import numpy as np


class AttributeInput():
    """An input to the BRB system.

    Consists of a set of antecedent attribute values and degrees of belief.

    Attributes:
        attr_input: X. Relates antecedent attributes with values and belief
        degrees. Must follow same order of reference values as in the model.
    """

    def __init__(self, attr_input: Dict[str, Dict[Any, float]]):
        self.attr_input = attr_input

    # TODO: add transformation methods

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
        (completely) activated.
    """

    def __init__(self, A_values: Dict[str, Any], beta: List[float],
                 delta: Dict[str, float] = None, theta: float = 1):
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

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.
        """
        self._assert_input(X)

        norm_delta = {attr: d / max(self.delta.values()) for attr, d
                      in self.delta.items()}
        weighted_alpha = [[
                alpha_i ** norm_delta[attr] for A_i, alpha_i
                in X.attr_input[attr].items() if A_i == self.A_values[attr]
            ] for attr in self.A_values.keys()
        ]

        return np.prod(weighted_alpha)

    def get_belief_degrees_complete(self, X: AttributeInput) -> Dict[Any, Any]:
        """Returns belief degrees transformed based on input completeness
        """
        self._assert_input(X)

        attribute_total_activations = {attr: sum(X.attr_input[attr].values())
                                       for attr in X.attr_input.keys()}

        rule_input_completeness = sum([attribute_total_activations[attr]
                                       for attr in self.A_values.keys()]) \
                                    / len(self.A_values.keys())

        norm_beta = [belief * rule_input_completeness for belief in self.beta]

        return norm_beta

    def _assert_input(self, X: AttributeInput):
        """Checks if `X` is proper.

        Guarantees that all the necessary attributes are present in X.
        """
        rule_attributes = set(self.A_values.keys())
        input_attributes = set(X.attr_input.keys())
        assert rule_attributes.intersection(input_attributes) == rule_attributes

class RuleBaseModel():
    """Parameters for the model.

    It contains the basic, standard information that will be used to manage the
    information and apply the operations.

    Attributes:
        U: Antecendent attributes' names.
        A: Referential values for each antecedent.
        D: Consequent referential values.
        F: ?
        rules: List of rules.
    """
    def __init__(self, U: List[str], A: Dict[str, List[Any]], D: List[Any], F=None):
        # no repeated elements for U
        assert len(U) == len(set(U))
        self.U = U

        # referential values for all antecedent attributes must be provided
        assert set(U) == set(A.keys())
        self.A = A

        self.D = D
        self.F = F

        self.rules = list()

    def add_rule(self, new_rule: Rule):
        """Adds a new rule to the model.

        Verifies if the given rule agrees with the model settings and adds it
        to `.rules`.
        """
        # all reference values must be related to an attribute
        assert set(new_rule.A_values.keys()) == set(self.U)

        # the reference values that activate the rule must be a valid
        # referential value in the self
        for U_i, A_i in new_rule.A_values.items():
            assert A_i in self.A[U_i]

        self.rules.append(new_rule)

    def run(self, X: AttributeInput):
        """Infer the output based on the RIMER approach.

        Args:
            X: Attribute's data to be fed to the rules.
        """
        # input for all valid antecedents must be provided
        for U_i in X.attr_input.keys():
            assert U_i in self.U

        # 2. matching degree
        # alphas[k] = \alpha_k = matching degree of k-th rule
        alphas = [rule.get_matching_degree(X) for rule in self.rules]

        # 3. activation weight
        total_theta_alpha = sum([rule.theta*alpha_k for rule, alpha_k
                                 in zip(self.rules, alphas)])
        # activation_weights[k] = w_k = activation weight of the k-th rule
        activation_weights = [rule.theta * alpha_k / total_theta_alpha
                              for rule, alpha_k in zip(self.rules, alphas)]

        # 4. analytical ER algorithm
        # the following is based on "Inference and learning methodology of
        # belief-rule-based expert system for pipeline leak detection" by
        # _Xu et al._
        belief_degrees = [rule.beta for rule in self.rules]
        total_belief_degrees = [sum(beta_k) for beta_k in belief_degrees]
        left_prods = list()
        # transpose belief degrees
        for rules_beta_j in map(list, zip(*belief_degrees)):
            left_prods.append(np.prod(
                [weight_k * rules_beta_jk + 1 - weight_k * total_belief_degrees_k  # pylint: disable=line-too-long
                 for weight_k, rules_beta_jk, total_belief_degrees_k
                 in zip(activation_weights, rules_beta_j, total_belief_degrees)]
            ))
        right_prod = np.prod(
            [1 - weight_k * total_belief_degrees_k
             for weight_k, total_belief_degrees_k
             in zip(activation_weights, total_belief_degrees)]
        )
        mi = 1 / (sum(left_prods) - (len(self.D) - 1) * right_prod)

        belief_degrees = [mi * (left_prod - right_prod) / (1 - mi * \
                          np.prod([1 - weight_k for weight_k
                                   in activation_weights]))
                          for left_prod in left_prods]

        # TODO: add utility calculation

        return belief_degrees
