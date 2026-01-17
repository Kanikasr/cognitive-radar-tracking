import numpy as np
from simulation.environment import TrackingEnvironment
from filters.ekf import ExtendedKalmanFilter
from decision.adaptive_policy import AdaptiveSensingPolicy
from learning.policy_model import LearnedSensingPolicy

# ------------------------
# LOAD LEARNED POLICY
# ------------------------
import os
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X = np.load(os.path.join(PROJECT_ROOT, "X.npy"))
y = np.load(os.path.join(PROJECT_ROOT, "y.npy"))

learned_policy = LearnedSensingPolicy()
learned_policy.train(X, y)

# ------------------------
# CONFIG
# ------------------------
TIMESTEPS = 50
RUNS = 10

LOW_COST = 1
HIGH_COST = 5
LOW_NOISE_STD = 3.0
HIGH_NOISE_STD = 0.1

def run_system(mode):
    errors = []
    total_cost = 0
    high_count = 0

    env = TrackingEnvironment()
    ekf = ExtendedKalmanFilter(dt=1.0)

    for t in range(TIMESTEPS):
        true_state = env.step()
        ekf.predict()

        uncertainty = ekf.P[0, 0] + ekf.P[1, 1]

        if mode == "FIXED":
            action = "LOW_QUALITY"

        elif mode == "RULE":
            action = "HIGH_QUALITY" if uncertainty > 8.0 else "LOW_QUALITY"

        elif mode == "LEARNED":
            action = learned_policy.select_action([uncertainty])

        # Cost + noise
        if action == "HIGH_QUALITY":
            ekf.set_measurement_noise(HIGH_NOISE_STD)
            total_cost += HIGH_COST
            high_count += 1
        else:
            ekf.set_measurement_noise(LOW_NOISE_STD)
            total_cost += LOW_COST

        measurement = env.observe(action)
        ekf.update(measurement)

        error = np.linalg.norm(ekf.x[:2] - true_state[:2])
        errors.append(error)

    return np.mean(errors), total_cost, high_count


# ------------------------
# RUN EVALUATION
# ------------------------
for mode in ["FIXED", "RULE", "LEARNED"]:
    err, cost, count = run_system(mode)
    print(f"{mode} POLICY")
    print("Mean tracking error:", round(err, 3))
    print("Total sensing cost:", cost)
    print("High-quality uses:", count)
    print("-" * 40)
