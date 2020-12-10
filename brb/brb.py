"""BRB RIMER implementation.

Based on "Belief Rule-Base Inference Methodology Using the Evidential Reasoning
Approach - RIMER", by _Yang et al._, this module implements three data
structures to construct expert systems based on the Evidential Reasoning
approach.

    Typical usage example:

    >>> from brb import RuleBaseModel, Rule, AttributeInput
    >>> model = RuleBaseModel(
    ...     U=['Antecedent'],
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
from copy import copy
from typing import List, Any, Dict, Union, Callable

from interval import interval
import numpy as np
import pandas as pd

from .antecedent import Antecedent, ContinuousAntecedent, DiscreteAntecedent, CategoricalAntecedent
from .attr_input import AttributeInput
from .rule import Rule

class RuleBaseModel():
    """Parameters for the model.

    It contains the basic, standard information that will be used to manage the
    information and apply the operations.

    Attributes:
        U: Antecendents.
        D: Consequent referential values.
        F: ?
        rules: List of rules.
    """
    def __init__(self, U: List[Antecedent], D: List[Any], F=None):
        # no repeated elements for U
        U_names = [U_i.name for U_i in U]
        assert len(U_names) == len(set(U_names))

        self.U = U

        self.D = D
        self.F = F

        self.rules = list()

    @property
    def U_names(self) -> List[str]:
        return [U_i.name for U_i in self.U]

    def get_antecedent_by_name(self, antecedent_name) -> Antecedent:
        for U_i in self.U:
            if U_i.name == antecedent_name:
                return U_i

        return None

    def add_rule(
            self,
            A_values: Dict[str, Any],
            beta: List[float],
            delta: Dict[str, float] = None,
            theta: float = 1,
            matching_degree: Union[str, callable] = 'arithmetic'
        ) -> Rule:
        """Adds a new rule to the model.

        Verifies if the given rule agrees with the model settings and adds it
        to `.rules`.
        """
        # all reference values must be related to an attribute
        assert set(A_values.keys()).issubset(set(self.U_names))

        # TODO: handle NaN values

        # consequent values must agree in shape with the model's consequents
        assert len(beta) == len(self.D)

        _A_values = {
            self.get_antecedent_by_name(U_i): A_value
            for U_i, A_value in A_values.items()
        }

        new_rule = Rule(
            A_values=_A_values,
            beta=beta,
            delta=delta,
            theta=theta,
            matching_degree=matching_degree
        )

        self.rules.append(new_rule)

        return new_rule

    def add_rules_from_df(
            self,
            rules_df: pd.DataFrame,
            thetas: str = None,
            delta_cols: List[str] = None,
        ):
        """Adds rules from pandas.DataFrame object. Columns must agree to model.

        Args:
            rules_df: Rules dataframe. Each row must be a rule. The columns must
            be the antecedents, consequents and rule weights (optional).
            thetas: Rules weights column. If `None` (default value), same weight
            (1.0) is given to all rules.
            delta_cols: Columns from `rules_df` containing the attribute
            weights for each rule.
        """
        # TODO: add deltas input support
        antecedents_df = rules_df[self.U_names]
        consequents_df = rules_df[self.D]

        A_ks = np.matrix(antecedents_df.values)
        betas = np.matrix(consequents_df.values)

        if thetas is not None:
            thetas = rules_df[thetas].values

        deltas = None
        # only valid if there are 1:1 weights to attributes
        if delta_cols is not None and len(delta_cols) == len(self.U):
            deltas = np.matrix(rules_df[delta_cols].values)

        self.add_rules_from_matrix(
            A_ks=A_ks,
            betas=betas,
            thetas=thetas,
            deltas=deltas
        )

    def add_rules_from_matrix(
            self,
            A_ks: np.matrix,
            betas: np.matrix,
            deltas: np.matrix = None,
            thetas: np.array = None
        ):
        """Adds several rules through the input matrices.

        Args:
            A_ks: Rules antecedents referential values matrix. Each row is a
            rule and the columns are the antecedents, so the matrix values must
            be the referential values according to the model.
            betas: Consequents belief degrees. Must follow the same order as the
            model definition.
            deltas: Attribute weights of the rules. Must have the same shape as
            A_ks. If `None` (default value), equal weight (1) is given for all
            attributes over all rules.
            thetas: Rules weights. If `None` (default value), same weight is
            given for all rules (1).
        """
        # TODO: add support for uncertainty and incompleteness in the rule
        # definition for both antecedents and consequents

        # the number of rules must be consistent
        assert A_ks.shape[0] == betas.shape[0]

        # every rule must comply to the amount of antecedent attributes
        assert A_ks.shape[1] == len(self.U)

        # there must be as many consequents as in the model definition
        assert betas.shape[1] == len(self.D)

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

        # convert nan values to 0
        betas = np.nan_to_num(betas)
        _ruleid = 1
        for A_k, beta_k, delta, theta in zip(A_ks, betas, deltas, thetas):

            # converst to dict and drops nan values
            A_k = np.asarray(A_k)[0]

            A_values = {U_i: A_k_value for U_i, A_k_value
                        in zip(self.U_names, A_k) if not pd.isna(A_k_value)}

            del_k = np.asarray(delta)[0]

            delta = {U_i: del_k_value for U_i, del_k_value
                        in zip(self.U_names, del_k) if not pd.isna(del_k_value)}

            # transforms referential value to rule shape
            rule_beta = np.asarray(beta_k)[0]

            self.add_rule(A_values=A_values, beta=rule_beta, delta=delta,
                               theta=theta)
            _ruleid += 1

    # TODO: add get_ function that returns the full rules matrix (all
    # combination of antecedent attributes' values) as a boilerplate for
    # defining the full set of rules.

    # TODO: add interface for "tunable" parameters

    def expand_rules(
            self,
            matching_method: Union[str, Callable] = None
        ) -> 'RuleBaseModel':
        """Expands rules with empty antecedents to cover all possibilities.

        Args:
            matching_method: Changes the matching degree calculation approach
            for the new rules. If `None`, keeps the current one.

        Returns:
            new_model: New model containing the original and the new rules.
        """
        # no previous antecedent => all rules are complete
        complete_rules = self.rules
        for U_i in self.U:
            # for the current antecedent, we don't know which rules are complete
            _rules = complete_rules
            complete_rules = list()

            for rule in _rules:
                if U_i.name not in rule.U_names:
                    complete_rules += rule.expand_antecedent(
                        U_i,
                        matching_method
                    )
                else:
                    new_rule = copy(rule)
                    new_rule.matching_degree = matching_method
                    complete_rules.append(new_rule)

        new_model = copy(self)

        new_model.rules = complete_rules

        return new_model

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

            assert U_i in self.U_names

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

def match_prefix(s: str, p: str):
    """Checks wether `p` is a prefix of `s`.
    """
    if p is None:
        return True

    s_p = s.lstrip()[:len(p)]

    return s_p == p

def csv2BRB(
        csv_filepath: str,
        antecedents_prefix: str,
        consequents_prefix: str,
        deltas_prefix: str = None,
        thetas: str = None,
    ) -> RuleBaseModel:
    """Converts csv table to a belief rule base (RuleBaseModel).

    The csv table must contain one column for each antecedent and one column for
    each consequent in pandas-friendly format. The table also must contain a
    header. The columns.names referred by `antecedent_cols` and
    `consequent_cols` will be used as the reference for the antecedents and
    consequents. Additionally, the weights of the rules and the attribute's
    weights of each rule can be provided in the same table.

    Args:
        csv_fillepath: csv filepath to the table containing the rules.
        antecedents_prefix: Prefix of the antecedents' columns names (as in the
        header of the table). Any column with this prefix will be understood as
        an antecedent.
        consequents_prefix: Prefix of the consequents' columns names (as in the
        header of the table). Any column with this prefix will be understood as
        an consequent.
        deltas_prefix: Prefix of the attribute weights' columns names (as in the
        header of the table). Exactly one column for each attribute must be
        provided. If `None` (default), will assign equal weight (1.0) to all
        attributes for every rule.
        thetas: Column name of the rules weights. If `None` (default), will
        assign equal weight (1.0) to all rules.

    Returns:
        model: Belief Rule Base containing all the rules defined in the csv
        file.
    """
    df_rules = pd.read_csv(csv_filepath)

    cols = df_rules.columns

    antecedent_cols = list()
    consequent_cols = list()
    delta_cols = list()
    for col in cols:
        if match_prefix(col, antecedents_prefix):
            antecedent_cols.append(col)
        elif match_prefix(col, consequents_prefix):
            consequent_cols.append(col)
        elif deltas_prefix is not None:
            if match_prefix(col, deltas_prefix):
                delta_cols.append(col)

    # define antecedents from columns
    U = list()
    for col in antecedent_cols:
        col_values = df_rules[col].dropna().values

        col_values = list(map(AttributeInput.prep_referential_value, col_values))

        extended_col_values = list()
        for col_value in col_values:
            if isinstance(col_value, dict):
                extended_col_values += list(col_value.keys())
            else:
                extended_col_values.append(col_value)

        col_values_types = list(map(type, extended_col_values))

        if str in col_values_types:
            U_i = CategoricalAntecedent(col, list(set(extended_col_values)))
        elif interval in col_values_types:
            U_i = ContinuousAntecedent(col)
        else:
            U_i = DiscreteAntecedent(col)

        U.append(U_i)

    model = RuleBaseModel(U=U, D=consequent_cols)

    model.add_rules_from_df(df_rules, thetas=thetas, delta_cols=delta_cols)

    return model
