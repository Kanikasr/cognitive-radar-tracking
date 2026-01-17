import numpy as np
from learning.policy_model import LearnedSensingPolicy

# Load data (for now, rerun generation inline)
X = np.load("X.npy")
y = np.load("y.npy")

policy = LearnedSensingPolicy()
policy.train(X, y)

print("Training complete")
