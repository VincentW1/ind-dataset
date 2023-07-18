import shapely.geometry as sg
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

class VRU():

	dt = 1.0/25.0
	dim_x = 4
	dim_z = 2

	def __init__(self, x, y, track_frame, type, trackId=0, scaling_factor=1):

		self.type = type
		self.trackId = trackId
		self.scaling_factor = scaling_factor
		self.miss_tracking_counter = 0

		self.update_position(x, y, track_frame)
		self.pos = [x, y]
		#TODO
		self.currently_tracked = False

		self.trackingX = None
		self.trackingY = None

		points = MerweScaledSigmaPoints(self.dim_x, alpha=.1, beta=2., kappa=-1)
		self.kf = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=self.dt, fx=f_cv, hx=h_cv, points=points)

		self.kf.x = np.array([0., 0., 0., 0.])  # initial state
		self.kf.P *= 500 # initial uncertainty
		q = Q_discrete_white_noise(dim=2, dt=self.dt, var=1)
		self.kf.Q[0:2, 0:2] = q
		self.kf.Q[2:4, 2:4] = q
		self.kf.R *= 0.1  # measurement uncertainty

		self.type = type


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
		z = np.array([x_measurement, y_measurement])  # your measurement
		self.kf.predict()
		self.kf.update(z)

		self.trackingX = self.kf.x[0]
		self.trackingY = self.kf.x[1]

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
		self.trackingY = self.kf.x[1]
		
		return self.trackingX, self.trackingY

	def increase_miss_tracking_counter(self):
		self.currently_tracked = False
		self.miss_tracking_counter += 1
