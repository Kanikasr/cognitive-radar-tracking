import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt

        # State estimate
        self.x = np.array([1.0, 1.0, 0.0, 0.0])

        # Covariance
        self.P = np.eye(4) * 10.0

        # Process noise
        self.Q = np.eye(4) * 0.1

        # Measurement noise (will be adaptive later)
        self.R = np.eye(2)

    def set_measurement_noise(self, noise_std):
        """
        Set measurement noise covariance based on sensing quality.
        """
        self.R = np.eye(2) * (noise_std ** 2)

    def predict(self):
        x, y, vx, vy = self.x

        # Motion model
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        x, y, _, _ = self.x

        # Predicted measurement
        h = np.array([
            np.sqrt(x**2 + y**2),
            np.arctan2(y, x)
        ])

        # Jacobian
        eps = 1e-5
        H = np.array([
            [x / (h[0] + eps), y / (h[0] + eps), 0, 0],
            [-y / (x**2 + y**2 + eps), x / (x**2 + y**2 + eps), 0, 0]
        ])

        y_residual = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.P = (np.eye(4) - K @ H) @ self.P
