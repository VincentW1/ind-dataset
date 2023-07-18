from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import approx_fprime
import numpy as np
import shapely.geometry as sg

class VRU():
    dt = 1.0/25.0
    dim_x = 6
    dim_z = 2

    def __init__(self, x, y, track_frame, type, trackId=0, scaling_factor=1):
        self.type = type
        self.trackId = trackId
        self.scaling_factor = scaling_factor
        self.miss_tracking_counter = 0

        self.update_position(x, y, track_frame)
        self.pos = [x, y]
        self.currently_tracked = False

        self.trackingX = None
        self.trackingY = None

        self.lastMeasurementX = None
        self.lastMeasurementY = None

        self.kf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.kf.x = np.array([x, 0., 0., y, 0., 0.])  # initial state
        self.kf.P *= 10  # initial uncertainty
        self.kf.Q = np.eye(self.dim_x)  # Initialize Q as an identity matrix
        q = Q_discrete_white_noise(dim=3, dt=self.dt, var=1)  # 3 state variables for 1D motion (position, velocity, acceleration)
        self.kf.Q[0:3, 0:3] = q * 200  # Increase x direction process noise
        self.kf.Q[3:6, 3:6] = q * 200  # Increase y direction process noise
        self.kf.R[0, 0] *= 0.5
        self.kf.R[1, 1] *= 0.5

        self.type = type

    def f(self, x, dt):
        """ state transition function """
        f_x = np.zeros_like(x)
        f_x[0] = x[0] + x[1] * dt + 0.5 * x[2] * dt**2  # x_position
        f_x[1] = x[1] + x[2] * dt  # x_velocity
        f_x[2] = x[2]  # x_acceleration
        f_x[3] = x[3] + x[4] * dt + 0.5 * x[5] * dt**2  # y_position
        f_x[4] = x[4] + x[5] * dt  # y_velocity
        f_x[5] = x[5]  # y_acceleration
        return f_x

    def h(self, x):
        """ measurement function """
        return np.array([x[0], x[3]])  # we only measure position

    def update_position(self, x, y, track_frame):
        self.x = x
        self.y = y
        self.pos = [x, y]
        self.circle = sg.Point(self.x, self.y).buffer(1)
        self.last_track_frame = track_frame

    def get_outer_poly(self):
        poly = sg.Polygon(self.circle)
        return poly

    def get_position(self):
        return self.x, self.y

    def update_tracking(self, x_measurement, y_measurement):
        self.lastMeasurementX = x_measurement
        self.lastMeasurementY = y_measurement
        self.miss_tracking_counter = 0
        self.currently_tracked = True
        z = np.array([x_measurement, y_measurement])  # your measurement
        epsilon = 1e-5  # step size for numerical differentiation
        fx = lambda x: self.f(x, self.dt)  # function of state only
        F = approx_fprime(self.kf.x, fx, epsilon)  # numerical Jacobian
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])  # Jacobian of the measurement function

        self.kf.predict()
        self.kf.update(z, HJacobian=lambda x: H, Hx=self.h)

        self.trackingX = self.kf.x[0]
        self.trackingY = self.kf.x[3]

        print("Current Tracking for VRU #" + str(self.trackId))
        print("MeasurementX: " + str(x_measurement))
        print("MeasurementY: " + str(y_measurement))
        print("GTX: " + str(self.x))
        print("GTY: " + str(self.y))
        print("TrackingX: " + str(self.trackingX))
        print("TrackingY: " + str(self.trackingY))

        return self.trackingX, self.trackingY

    def predict_tracking(self):
        print("ONLY PREDICTING NO LOS")
        self.kf.predict()
        self.trackingX = self.kf.x[0]
        self.trackingY = self.kf.x[3]

        return self.trackingX, self.trackingY

    def increase_miss_tracking_counter(self):
        self.currently_tracked = False
        self.miss_tracking_counter += 1
