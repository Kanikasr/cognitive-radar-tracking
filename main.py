import numpy as np
from simulation.environment import TrackingEnvironment
from filters.ekf import ExtendedKalmanFilter
from learning.policy_model import LearnedSensingPolicy

# ------------------------
# LOAD DATA
# ------------------------
X = np.load("X.npy")
y = np.load("y.npy")

# ------------------------
# TRAIN LEARNED POLICY
# ------------------------
learned_policy = LearnedSensingPolicy()
learned_policy.train(X, y)

print("✅ Learned policy trained")

# ------------------------
# RUN SYSTEM
# ------------------------
env = TrackingEnvironment()
ekf = ExtendedKalmanFilter(dt=1.0)

LOW_COST = 1
HIGH_COST = 5
LOW_NOISE_STD = 3.0
HIGH_NOISE_STD = 0.1

total_cost = 0

for t in range(30):
    true_state = env.step()
    ekf.predict()

    # Feature
    uncertainty = ekf.P[0, 0] + ekf.P[1, 1]

    # ML decision
    action = learned_policy.select_action([uncertainty])

    # Apply sensing effect
    if action == "HIGH_QUALITY":
        ekf.set_measurement_noise(HIGH_NOISE_STD)
        cost = HIGH_COST
    else:
        ekf.set_measurement_noise(LOW_NOISE_STD)
        cost = LOW_COST

    total_cost += cost

    measurement = env.observe(action)
    ekf.update(measurement)

    # Logging
    print(f"Time {t}")
    print("Action:", action)
    print("True position:", true_state[:2])
    print("Estimated position:", ekf.x[:2])
    print("Uncertainty:", round(uncertainty, 2))
    print("Total sensing cost:", total_cost)
    print("-" * 50)
