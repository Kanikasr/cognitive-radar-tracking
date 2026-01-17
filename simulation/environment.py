import numpy as np

class TrackingEnvironment:
    def __init__(self, dt=1.0, process_noise_std = 0.5, measurement_noise_std=1.0):
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.state = np.array([0.0, 0.0, 1.0, 0.5])

    def step(self):
        x, y, vx, vy = self.state
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt

        vx_new = vx + np.random.randn() * self.process_noise_std
        vy_new = vy + np.random.randn() * self.process_noise_std

        self.state = np.array([x_new, y_new, vx_new, vy_new])
        return self.state.copy()

    def observe(self, sensing_mode="LOW_QUALITY"):
        x, y, _, _ = self.state
        range_meas = np.sqrt(x**2 + y**2)
        bearing_meas = np.arctan2(y, x)

        noise_scale = 0.3 if sensing_mode == "HIGH_QUALITY" else 1.0
        noise = np.random.randn(2) * self.measurement_noise_std * noise_scale
        return np.array([range_meas, bearing_meas]) + noise
