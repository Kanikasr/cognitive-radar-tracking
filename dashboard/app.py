# ===============================
# Adaptive Tracking Dashboard
# ===============================

import sys
import os

# -------------------------------
# FIX IMPORT PATH
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulation.environment import TrackingEnvironment
from filters.ekf import ExtendedKalmanFilter
from learning.policy_model import LearnedSensingPolicy

# -------------------------------
# LOAD TRAINING DATA
# -------------------------------
X = np.load(os.path.join(PROJECT_ROOT, "X.npy"))
y = np.load(os.path.join(PROJECT_ROOT, "y.npy"))

# -------------------------------
# FREEZE TRAINED POLICY
# -------------------------------
@st.cache_resource
def load_policy():
    policy = LearnedSensingPolicy()
    policy.train(X, y)
    return policy

policy = load_policy()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Adaptive Tracking Dashboard", layout="wide")
st.title("Adaptive Sequential Decision System")

st.markdown("""
This dashboard demonstrates **state estimation under uncertainty**
with a **learning-based adaptive sensing policy**.
""")

# -------------------------------
# CONTROLS
# -------------------------------
steps = st.slider("Simulation steps", 10, 100, 30)
process_noise = st.slider("Process noise", 0.1, 1.0, 0.1)
demo_mode = st.checkbox("Demo mode (force adaptive switching)")

# -------------------------------
# RUN SIMULATION
# -------------------------------
env = TrackingEnvironment(process_noise_std=process_noise)
ekf = ExtendedKalmanFilter(dt=1.0)

true_positions = []
est_positions = []
uncertainties = []
actions = []
scores = []

total_cost = 0
LOW_COST = 1
HIGH_COST = 5

for _ in range(steps):
    true_state = env.step()
    ekf.predict()

    # Raw EKF uncertainty
    raw_uncertainty = ekf.P[0, 0] + ekf.P[1, 1]

    # Scale uncertainty into training regime (dashboard-only)
    scaled_uncertainty = raw_uncertainty * 50

    # Model confidence
    prob_high = policy.model.predict_proba([[scaled_uncertainty]])[0][1]

    # Decision logic
    if demo_mode and scaled_uncertainty > 300:
        action = "HIGH_QUALITY"
    else:
        action = "HIGH_QUALITY" if prob_high > 0.5 else "LOW_QUALITY"

    # Cost
    total_cost += HIGH_COST if action == "HIGH_QUALITY" else LOW_COST

    # Measurement and update
    measurement = env.observe(action)
    ekf.update(measurement)

    # Store data
    true_positions.append(true_state[:2])
    est_positions.append(ekf.x[:2])
    uncertainties.append(raw_uncertainty)
    actions.append(action)
    scores.append(prob_high)

true_positions = np.array(true_positions)
est_positions = np.array(est_positions)

# -------------------------------
# PLOTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("True vs Estimated Position")
    fig, ax = plt.subplots()
    ax.plot(true_positions[:, 0], true_positions[:, 1], label="True")
    ax.plot(est_positions[:, 0], est_positions[:, 1], label="Estimated")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Uncertainty Over Time")
    fig, ax = plt.subplots()
    ax.plot(uncertainties)
    ax.set_ylabel("Uncertainty")
    st.pyplot(fig)

st.subheader("Policy Confidence (P[HIGH_QUALITY])")
st.line_chart(scores)

st.subheader("Decisions & Cost")
st.write(f"**Total sensing cost:** {total_cost}")
st.write("Actions taken:")
st.write(actions)
