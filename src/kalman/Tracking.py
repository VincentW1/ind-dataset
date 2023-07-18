from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np

dt = 0.1
dim_x = 4
dim_z = 2

points = MerweScaledSigmaPoints(dim_x, alpha=.1, beta=2., kappa=-1)

def f_cv(x, dt):
    """ state transition function for a constant velocity aircraft"""
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]], dtype=float)
    return np.dot(F, x)

def h_cv(x):
    return x[:2]


def add_sensor_noise(x_true, y_true, sigma_x, sigma_y):
    """
    Function to add Gaussian noise to a 2D ground truth measurement.

    Args:
    x_true: Ground truth x-coordinate
    y_true: Ground truth y-coordinate
    sigma_x: Standard deviation of the noise in the x direction
    sigma_y: Standard deviation of the noise in the y direction

    Returns:
    x_noisy: Noisy x-coordinate
    y_noisy: Noisy y-coordinate
    """

    # Add Gaussian noise
    x_noisy = x_true + np.random.normal(0, sigma_x)
    y_noisy = y_true + np.random.normal(0, sigma_y)

    return x_noisy, y_noisy


class Tracking:


    def __init__(self, trackingId, gtX, gtY):
        self.trackingId = trackingId
        self.gtX = gtX
        self.gtY = gtY

        kf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=f_cv, hx=h_cv, points=points)
        kf.x = np.array([0., 0., 4., 3.])  # initial state
        kf.P *= 0.2  # initial uncertainty
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
        kf.Q[0:2, 0:2] = q
        kf.Q[2:4, 2:4] = q
        kf.R *= 1  # measurement uncertainty


    def update_tracking(self, new_trackingX, new_trackingY):



        self.trackingX = new_trackingX
        self.trackingY = new_trackingY

    def update(self, new_gtX, new_gtY):
        self.gtX = new_gtX
        self.gtY = new_gtY


    def get_tracking_id(self):
        return self.trackingId

    def get_tracking_coordinates(self):
        return self.trackingX, self.trackingY

    def get_ground_truth_coordinates(self):
        return self.gtX, self.gtY
