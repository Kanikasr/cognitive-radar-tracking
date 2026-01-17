# Cognitive Radar for Target Detection & Tracking

This repository documents my M.Tech thesis work on cognitive radar systems,
focused on adaptive target tracking under noisy sensing conditions.

## Problem Overview
Traditional radar systems operate with fixed parameters and struggle in
non-stationary environments. This work explores a cognitive radar framework
where sensing and action are coupled through feedback to improve tracking
accuracy over time.

## System Design
<img width="749" height="503" alt="image" src="https://github.com/user-attachments/assets/a79132b4-7bc8-490c-bf0a-2e827ed16008" />


The system is built around a perception–action cycle:
- Receiver estimates target state using Bayesian filtering
- Transmitter adapts waveform parameters based on receiver feedback
- Feedback loop minimizes tracking error iteratively

## Methods Used
- State-space modeling of target dynamics
- EKF, UKF, and Cubature Kalman Filter for nonlinear tracking
- Dynamic optimization of waveform parameters
- RMSE-based evaluation over multiple iterations
<img width="900" height="523" alt="image" src="https://github.com/user-attachments/assets/4236980d-4e50-491a-a338-bc2ed7374b94" />

## Results
Experiments demonstrate consistent reduction in tracking error across cycles,
with Cubature Kalman Filter showing improved stability in highly nonlinear
conditions.

