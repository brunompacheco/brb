from typing import List, Dict, Any


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
        self.U = U
        self.A = A
        self.D = D
        self.F = F

        self.rules = list()

    def add_rule(new_rule: Rule):
        # TODO: remove model parameter from Rule and AttributeInput
        # TODO: move assertions from Rule.__init__() here
        self.rules.append(new_rule)

class Rule():
    """A rule definition in a BRB system.

    It translates expert knowledge into a mapping between the antecedents and the consequents.

    Attributes:
        model: The rule-base model to which this rule should apply to.
        A_values: A^k. Reference values for each antecedent attribute that activate the rule.
        delta: \delta_k. Relative weights of antecedent attributes.
        theta: \theta_k. Rule weight.
        beta: \bar{\beta}. Expected belief degrees of consequents if rule is (completely) activated.
    """

    def __init__(self, model: RuleBaseModel, A_values: Dict[str, Any], delta: Dict[str, float], theta: float, beta: Dict[Any, Any]):
        self.model = model

        # all reference values must be related to an attribute
        assert A_values.keys() in model.U

        # the reference values that activate the rule must be a valid referential value in the model
        for U_i, A_i in A_values.items():
            assert A_i in model.A[U_i]
        self.A_values = A_values

        # there must exist a weight for all antecedent attributes that activate the rule
        for U_i in A_values.keys():
            assert U_i in delta.keys()
        self.delta = delta

        self.theta = theta

        # the belief degrees must be applied to values defined by the model
        for D_n in beta.keys():
            assert D_n in model.D
        self.beta = beta

class AttributeInput():
    """An input to the BRB system.

    Consists of a set of antecedent attribute values and degrees of belief.

    Attributes:
        attr_input: A^*. Relates antecedent attributes with values and belief degrees.
    """

    def __init__(self, model, attr_input: Dict[str, Tuple[Any, float]]):
        self.model = model

        # it must provide input for valid antecedents
        for U_i in attr_input.keys():
            assert U_i in model.U
        self.attr_input = attr_input
