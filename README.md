# Adaptive Sequential Decision System

This project demonstrates **adaptive sequential decision-making under uncertainty**
using classical state estimation and a lightweight learning-based policy.

The system estimates hidden states from noisy observations and adaptively
controls sensing quality based on uncertainty and cost trade-offs.

---

## Motivation

In many real-world systems, the true state of the environment is not directly
observable. Decisions must be made sequentially using noisy measurements,
while balancing performance and operational cost.

This project focuses on:
- Estimating hidden states over time
- Quantifying uncertainty
- Using uncertainty to drive adaptive decisions
- Learning when expensive sensing is justified

---

## Core Concepts

- Partial observability
- State-space modeling
- Sequential Bayesian estimation
- Uncertainty propagation
- Adaptive sensing and control
- Learning-based decision policies

---

## System Overview

The system consists of five main components:

### 1. Environment Simulation
- Simulates a dynamic system with hidden states (position and velocity)
- Adds process noise to model uncertainty in dynamics
- Produces noisy, nonlinear observations

### 2. State Estimation (EKF)
- Uses an Extended Kalman Filter
- Performs prediction and update steps sequentially
- Tracks estimation uncertainty via covariance propagation

### 3. Adaptive Decision Logic
- Chooses between LOW and HIGH quality sensing
- Balances estimation accuracy against sensing cost
- Forms a closed feedback loop with the estimator

### 4. Learning Component
- Trains a lightweight classifier on uncertainty features
- Learns when high-quality sensing is necessary
- Generalizes beyond fixed or rule-based strategies

### 5. Deployment Dashboard
- Interactive Streamlit application
- Visualizes trajectories, uncertainty, and decisions
- Includes demo mode for adaptive switching visualization

---

## Project Structure

adaptive_tracking_system/
│
├── dashboard/          # Streamlit deployment app

├── simulation/         # Environment & dynamics

├── filters/            # EKF implementation

├── decision/           # Adaptive sensing logic

├── learning/           # Learned policy

├── evaluation/         # Policy comparison & metrics

├── main.py             # Training / inference entry point

├── README.md

└── .gitignore

---

## How to Run

### 1. Install dependencies

pip install numpy scipy scikit-learn matplotlib streamlit

### 2. Run training / inference

python main.py

### 3. Launch dashboard

streamlit run dashboard/app.py

---

## Evaluation

The project supports comparison between:
- Fixed sensing policy
- Rule-based adaptive policy
- Learning-based adaptive policy

Evaluation metrics include:
- Mean tracking error
- Uncertainty evolution
- Total sensing cost

---

## Key Insight

The learned policy does not blindly adapt.
It increases sensing quality only when uncertainty justifies the additional cost.

In nominal conditions, the system conserves resources while maintaining
stable tracking performance.

---

## Notes

- Uses simulated data only
- No real-world or proprietary datasets
- Designed for ML / Data Science audiences
- Avoids defense-specific jargon


