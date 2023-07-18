from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np

def f_cv(x, dt):
    """ state transition function for a constant velocity aircraft"""
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]], dtype=float)
    return np.dot(F, x)

def h_cv(x):
    return x[:2]

dt = 1.0
dim_x = 4
dim_z = 2

# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(dim_x, alpha=.1, beta=2., kappa=-1)

kf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=f_cv, hx=h_cv, points=points)
kf.x = np.array([0., 0., 4., 3.])  # initial state
kf.P *= 0.2  # initial uncertainty
q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
kf.Q[0:2, 0:2] = q
kf.Q[2:4, 2:4] = q
kf.R *= 1  # measurement uncertainty

z = np.array([4., 3.])  # your measurement
kf.predict()
kf.update(z)

print(kf.x)