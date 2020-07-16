from typing import List, Dict, Tuple, Any

import numpy as np


class AttributeInput():
    """An input to the BRB system.

    Consists of a set of antecedent attribute values and degrees of belief.

    Attributes:
        attr_input: X. Relates antecedent attributes with values and belief degrees. Must follow same order of reference values as in the model.
    """

    def __init__(self, attr_input: Dict[str, Dict[Any, float]]):
        self.attr_input = attr_input

    # TODO: add transformation methods

class Rule():
    """A rule definition in a BRB system.

    It translates expert knowledge into a mapping between the antecedents and the consequents. We assume that it is defined as a pure AND rules, that is, the only logical relation between the input attributes is the AND function.

    Attributes:
        A_values: A^k. Dictionary that matches reference values for each antecedent attribute that activates the rule.
        delta: \delta_k. Relative weights of antecedent attributes.
        theta: \theta_k. Rule weight.
        beta: \bar{\beta}. Expected belief degrees of consequents if rule is (completely) activated.
    """

    def __init__(self, A_values: Dict[str, Any], delta: Dict[str, float], theta: float, beta: List[float]):
        self.A_values = A_values

        # there must exist a weight for all antecedent attributes that activate the rule
        for U_i in A_values.keys():
            assert U_i in delta.keys()
        self.delta = delta

        self.theta = theta
        self.beta = beta

    def get_matching_degree(self, X: AttributeInput) -> float:
        """Calculates the matching degree of the rule based on input `X`.
        """
        self._assert_input(X)

        norm_delta = {attr: d / max(self.delta.values()) for attr, d in self.delta.items()}
        weighted_alpha = [[alpha_i ** norm_delta[attr] for A_i, alpha_i in X.attr_input[attr].items() if A_i == self.A_values[attr]] for attr in self.A_values.keys()]

        return np.prod(weighted_alpha)

    def get_belief_degrees_complete(self, X: AttributeInput) -> Dict[Any, Any]:
        """Returns belief degress transformed based on input completeness
        """
        self._assert_input(X)

        attribute_total_activations = {attr: sum(X.attr_input[attr].values()) for attr in X.attr_input.keys()}

        rule_input_completeness = sum([attribute_total_activations[attr] for attr in self.A_values.keys()]) / len(self.A_values.keys())

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
    
    It contains the basic, standard information that will be used to manage the information and apply the operations.

    Attributes:
        U: Antecendent attributes' names.
        A: Referential values for each antecedent.
        D: Consequent referential values.
        F: ?
        rules: List of rules.
    """
    def __init__(self, U: List[str], A: Dict[str, List[Any]], D: List[Any], F):
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

        Verifies if the given rule agrees with the model settings and adds it to `.rules`.
        """
        # all reference values must be related to an attribute
        assert set(new_rule.A_values.keys()) == set(self.U)

        # the reference values that activate the rule must be a valid referential value in the self
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
        total_theta_alpha = sum([rule.theta*alpha_k for rule, alpha_k in zip(self.rules, alphas)])
        # activation_weights[k] = w_k = activation weight of the k-th rule
        activation_weights = [rule.theta * alpha_k / total_theta_alpha for rule, alpha_k in zip(self.rules, alphas)]

        # 4. degrees of belief
        # normalized_belief_degrees[k][j] = \beta_jk = normalized belief degree for k-th rule, j-th consequent
        normalized_belief_degrees = [rule.get_belief_degrees_complete(X) for rule in self.rules]

        # 5. probability masses
        # probability_masses[k][j] = m_jk = probability mass of k-th rule, j-th consequent
        probability_masses = [rule_weight * rule_belief_degrees for rule_weight, rule_belief_degrees in zip(activation_weights, normalized_belief_degrees)]
        # unassigned_masses[k] = m_Dk = unassigned probability mass of k-th rule
        unassigned_masses = [1 - sum(probability_masses_k) for probability_masses_k in probability_masses]
        # relative_importance_masses[k] = \bar{m}_Dk = relative importance probability mass of k-th rule
        relative_importance_masses = [1 - weight for weight in activation_weights]
        # incompleteness_masses[k] = \tilde{m}_Dk = incompleteness probablity mass of k-th rule
        incompleteness_masses = [weight_k * (1 - sum(belief_degrees_k)) for weight_k, belief_degrees_k in zip(activation_weights, normalized_belief_degrees)]

        # this must be true by definition
        assert unassigned_masses == [bar_m + tilde_m for bar_m, tilde_m in zip(relative_importance_masses, incompleteness_masses)]

        # 6. ER algorithm
        from itertools import product
        m_I = probability_masses[0]
        importance_m_I = relative_importance_masses[0]
        incompleteness_m_I = incompleteness_masses[0]
        for k in range(len(self.rules) - 1):
            K_I = 1 / (1 - [a + b for a, b in product(m_I, probability_masses[k+1])])

            importance_m_I = K_I * (importance_m_I * relative_importance_masses[k+1])
            incompleteness_m_I = K_I * (incompleteness_m_I * incompleteness_masses[k+1] + incompleteness_m_I * relative_importance_masses[k+1] + importance_m_I * incompleteness_masses[k+1])
            m_DI = importance_m_I + incompleteness_m_I
            m_I = [K_I * (m_jI * m_j + m_jI * m_DI + m_DI * m_j) for m_jI, m_j in zip(m_I, probability_masses[k+1])]

        belief_degrees = [m_jI / (1 - importance_m_I) for m_jI in m_I]
        remaining_belief_degree = incompleteness_m_I / (1 - importance_m_I)

        # TODO: add utility calculation

        return belief_degrees, remaining_belief_degree
