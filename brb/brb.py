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
from ast import literal_eval
from typing import List, Dict, Any, Union, Callable
from warnings import warn

import numpy as np
import pandas as pd

from interval import interval


def _prep_referential_value(X_i):
    """Converts pythonic string input to acceptable data type.
    """
    try:
        # TODO: rewrite code based on `literal_eval` application on the input
        eval_X_i = literal_eval(X_i)

        # interval strings could be conflicting with lists
        if type(eval_X_i) is not list:
            _X_i = eval_X_i
        else:
            _X_i = X_i
    except ValueError:
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

        if _check_is_interval(_X_i):
            _X_i_start, _X_i_end = _prep_interval(_X_i)

            try:
                _X_i_start = int(_X_i_start)
                _X_i_end = int(_X_i_end)
            except ValueError:
                _X_i_start = float(_X_i_start)
                _X_i_end = float(_X_i_end)

            if isinstance(_X_i_start, int) and isinstance(_X_i_end, int):
                _X_i = set(range(_X_i_start, _X_i_end + 1))
            else:
                _X_i = interval[_X_i_start, _X_i_end]

    return _X_i

def _prep_interval(value: str):
    """Prepares interval input string to be converted to interval.interval.
    """
    if _check_is_interval(value):
        _value = value.replace(' ', '').replace('[', '').replace(']', '')
        start, end = _value.split(',')

        return start, end
    else:
        return value

def _check_is_interval(value: str):
    """Checks if `value` is properly to be converted to interval.interval.
    """
    is_interval = True

    _value = value.replace(' ', '')

    # intervals must start and end with brackets
    try:
        is_interval &= _value[0] == '['
        is_interval &= _value[-1] == ']'
    except IndexError:
        is_interval = False

    # intervals must have one (and only one) comma separating the boundaries
    n_commas = len(_value) - len(_value.replace(',', ''))
    is_interval &= n_commas == 1

    if ',' in _value:
        # an interval's boundaries must be numeric
        start, end = _value[1:-1].split(',')

        is_interval &= is_numeric(start)
        is_interval &= is_numeric(end)
    else:
        is_interval = False

    return is_interval

def is_numeric(a):
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

    @staticmethod
    def get_antecedent_matching(A_i, X_i) -> float:
        """Quantifies matching of an input and a referential value.

        Args:
            A_i: Referential value for antecedent U_i. Can be a category
            (string), continuous or discrete numerical value.
            X_i: Input value for antecedent U_i. Must be either a single value
            that matches the Referential value or a dictionary that maps the
            values to certainty.

        Returns:
            match: Between 0-1, quantifies how much `X_i` matches the
            referential value `A_i`.
        """
        match = 0.0

        _X_i = _prep_referential_value(X_i)
        _A_i = _prep_referential_value(A_i)

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
            warn('Input {} mismatches the referential value {}'.format(X_i, A_i))

        return match

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.

        Implementation based on the RIMER approach as proposed by _Yang et al._
        in "Belief rule-base inference methodology using the evidential
        reasoning Approach-RIMER", specifically eq. (6a).
        """
        self._assert_input(X)

        alphas_i = {
            U_i: self.get_antecedent_matching(
                self.A_values[U_i], X.attr_input[U_i]
            )
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
        D: Consequent referential values.
        F: ?
        rules: List of rules.
    """
    def __init__(self, U: List[str], D: List[Any], F=None):
        # no repeated elements for U
        assert len(U) == len(set(U))
        self.U = U
        # TODO: add antecedents types

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
