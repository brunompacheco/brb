"""BRB RIMER implementation.

Based on "Belief Rule-Base Inference Methodology Using the Evidential Reasoning
Approach - RIMER", by _Yang et al._, this module implements three data
structures to construct expert systems based on the Evidential Reasoning
approach.

    Typical usage example:

    >>> from brb.brb import RuleBaseModel, Rule, AttributeInput
    >>> model = RuleBaseModel(
    ...     U=['Antecedent'],
    ...     A={'Antecedent': ['good', 'bad']},
    ...     D=['good', 'bad']
    ... )
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
from typing import List, Dict, Any, Union, Callable

import numpy as np
import pandas as pd


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
        matching_degree: \phi. Defines how to calculate the matching degree for
        the rule. If `Callable`, must be a function that takes `delta`,
        `A_values` and `X` as input. If string, must be either 'geometric'
        (default) or 'arithmetic', which apply the respective weighted means.
    """

    def __init__(
            self,
            A_values: Dict[str,
            Any],
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

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (6a).
        """
        self._assert_input(X)

        if self.matching_degree == 'geometric':
            return self._geometric_matching_degree(self.delta, self.A_values, X)
        elif self.matching_degree == 'arithmetic':
            return self._arithmetic_matching_degree(self.delta, self.A_values, X)
        elif callable(self.matching_degree):
            return self.matching_degree(self.delta, self.A_values, X)

    @staticmethod
    def _arithmetic_matching_degree(
            delta: Dict[str, float],
            A_values: Dict[str, Any],
            X: AttributeInput
        ) -> float:
        norm_delta = {attr: d / sum(delta.values()) for attr, d
                      in delta.items()}
        weighted_alpha = [[
                alpha_i * norm_delta[attr] for A_i, alpha_i
                in X.attr_input[attr].items() if A_i == A_values[attr]
            ] for attr in A_values.keys()
        ]

        return np.sum(weighted_alpha)

    @staticmethod
    def _geometric_matching_degree(
            delta: Dict[str, float],
            A_values: Dict[str, Any],
            X: AttributeInput
        ) -> float:
        norm_delta = {attr: d / max(delta.values()) for attr, d
                      in delta.items()}
        weighted_alpha = [[
                alpha_i ** norm_delta[attr] for A_i, alpha_i
                in X.attr_input[attr].items() if A_i == A_values[attr]
            ] for attr in A_values.keys()
        ]

        return np.prod(weighted_alpha)

    def get_belief_degrees_complete(self, X: AttributeInput) -> Dict[Any, Any]:
        """Returns belief degrees transformed based on input completeness

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (8).
        """
        self._assert_input(X)

        # sum of activations of the referential values for each antecedent
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

    def __str__(self):
        A_values_str = ["({}:{})".format(U_i, A_i)
                        for U_i, A_i
                        in self.A_values.items()]

        str_out = r' /\ '.join(A_values_str)

        # TODO: add consequents labels
        str_out += ' => ' + str(self.beta)

        return str_out

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
    def __init__(self, U: List[str], A: Dict[str, List[Any]], D: List[Any],
                 F=None):
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
        assert set(new_rule.A_values.keys()).issubset(set(self.U))

        # the reference values that activate the rule must be a valid
        # referential value in the self
        for U_i, A_i in new_rule.A_values.items():
            assert A_i in self.A[U_i]

        # TODO: handle NaN values

        # consequent values must agree in shapep with the model's consequents
        assert len(new_rule.beta) == len(self.D)

        self.rules.append(new_rule)

    def add_rules_from_matrix(self, A_ks: np.matrix, betas: List[Any],
                              deltas: np.matrix = None, thetas: np.array = None):
        """Adds several rules through the input matrices.

        Args:
            A_ks: Rules antecedents referential values matrix. Each row is a
            rule and the columns are the antecedents, so the matrix values must
            be the referential values according to the model.
            betas: Consequent referential values.
            deltas: Attribute weights of the rules. Must have the same shape as
            A_ks. If `None` (default value), equal weight (1) is given for all
            attributes over all rules.
            thetas: Rules weights. If `None` (default value), same weight is
            given for all rules (1).
        """
        # TODO: add support for uncertainty and incompleteness in the rule
        # definition for both antecedents and consequents

        # the number of rules must be consistent
        assert A_ks.shape[0] == len(betas)

        # every rule must comply to the amount of antecedent attributes
        assert A_ks.shape[1] == len(self.U)

        # the values in the matrix must comply to the referential values in the
        # model
        for A_i, A_ref in zip(A_ks.T, self.A.values()):
            A_i = A_i[~pd.isna(A_i.tolist())]  # drops nan values
            assert np.isin(A_i, A_ref).all()

        # same is true for the consequents
        assert np.isin(betas, self.D).all()

        if deltas is None:
            # all antecedents have the same weight
            deltas = A_ks.shape[0] * [None, ]

        # there must be a weight for every antecedent for every rule
        assert len(deltas) == A_ks.shape[0]
        # TODO: deltas' rows must either be same size as A_ks' or None

        if thetas is None:
            # all rules have the sae weight
            thetas = np.ones(A_ks.shape[0])

        # there must be a weight for every rule
        assert len(thetas) == A_ks.shape[0]

        # beta values must agree with the model's referential values
        assert set(betas).issubset(self.D)

        for A_k, beta, delta, theta in zip(A_ks, betas, deltas, thetas):
            # converst to dict and drops nan values
            A_k = np.asarray(A_k)[0]
            A_values = {U_i: A_k_value for U_i, A_k_value
                        in zip(self.U, A_k) if not pd.isna(A_k_value)}

            # transforms referential value to rule shape
            rule_beta = {D_i: 0.0 for D_i in self.D}
            rule_beta[beta] = 1.0
            rule_beta = list(rule_beta.values())

            self.add_rule(Rule(A_values=A_values, beta=rule_beta, delta=delta,
                               theta=theta))

    # TODO: add get_ function that returns the full rules matrix (all
    # combination of antecedent attributes' values) as a boilerplate for
    # defining the full set of rules.

    # TODO: add interface for "tunable" parameters

    def run(self, X: AttributeInput):
        """Infer the output based on the RIMER approach.

        Based on the approach of "Inference and learning methodology of
        belief-rule-based expert system for pipeline leak detection" by
        _Xu et al._. Matching degrees and activation weights for all the rules
        are calculated first. An addition done in this implementation is the
        normalization of belief degrees once it was noticed that without this,
        the incompleteness in the input wouldn't be reflected in the resulting
        belief degrees. Finally, the final belief degrees are calculated from
        the analytical ER algorithm.

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
        # implementation based on eq. (7) of "Belief rule-base inference
        # methodology using the evidential reasoning Approach-RIMER", by
        # _Yang et al._

        # total_theta_alpha is the sum on the denominator of said equation
        total_theta_alpha = sum([rule.theta*alpha_k for rule, alpha_k
                                 in zip(self.rules, alphas)])
        total_theta_alpha = total_theta_alpha if total_theta_alpha != 0 else 1

        # activation_weights[k] = w_k = activation weight of the k-th rule
        activation_weights = [rule.theta * alpha_k / total_theta_alpha
                              for rule, alpha_k in zip(self.rules, alphas)]

        # 4. degrees of belief
        # use normalized belief degrees to compensate for incompleteness
        belief_degrees = [rule.get_belief_degrees_complete(X) for rule
                          in self.rules]

        # 5. analytical ER algorithm

        # sum of all belief degrees over the rules
        total_belief_degrees = [sum(beta_k) for beta_k in belief_degrees]

        # left_prods is the productory that appears both in the left-side
        # of the numerator of eq. (4) and in \mu. Note that this depends on j
        left_prods = list()
        # map is a transposition of belief_degrees
        for rules_beta_j in map(list, zip(*belief_degrees)):
            left_prods.append(np.prod(
                [weight_k * rules_beta_jk + 1 - weight_k * total_belief_degrees_k  # pylint: disable=line-too-long
                 for weight_k, rules_beta_jk, total_belief_degrees_k
                 in zip(activation_weights, rules_beta_j, total_belief_degrees)]
            ))
        # the productory that appears in the right side of the numerator of eq.
        # (4) and in \mu
        right_prod = np.prod(
            [1 - weight_k * total_belief_degrees_k
             for weight_k, total_belief_degrees_k
             in zip(activation_weights, total_belief_degrees)]
        )
        mu = 1 / (sum(left_prods) - (len(self.D) - 1) * right_prod)

        # eq. (4)
        belief_degrees = [mu * (left_prod - right_prod) / (1 - mu * \
                          np.prod([1 - weight_k for weight_k
                                   in activation_weights]))
                          for left_prod in left_prods]
        # TODO: `brb.py:307: RuntimeWarning: invalid value encountered in
        # double_scalars` while running test.py

        # handles the case where there is 0 certainty, i.e., completely 0 input
        if all(np.isnan(belief_degrees)):
            belief_degrees = [mu * (left_prod - right_prod) for left_prod
                              in left_prods]

        # TODO: add utility calculation

        return belief_degrees
