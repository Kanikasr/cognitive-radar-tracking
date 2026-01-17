class AdaptiveSensingPolicy:
    """
    Rule-based adaptive sensing policy.
    Designed to be non-degenerate (used as teacher for learning).
    """

    def __init__(self, uncertainty_threshold=1200.0):
        self.uncertainty_threshold = uncertainty_threshold
        self.low_cost = 1
        self.high_cost = 5

    def select_action(self, covariance):
        position_uncertainty = covariance[0, 0] + covariance[1, 1]

        if position_uncertainty > self.uncertainty_threshold:
            return "HIGH_QUALITY"
        else:
            return "LOW_QUALITY"

    def get_cost(self, action):
        return self.high_cost if action == "HIGH_QUALITY" else self.low_cost
