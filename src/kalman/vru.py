import shapely.geometry as sg

class VRU():

	def __init__(self, x, y, type, trackId=0, scaling_factor=1):

		self.type = type
		self.trackId = trackId
		self.scaling_factor = scaling_factor

		self.update_position(x, y)

		self.in_critical_range = 0
		self.risk_counter = 0
		#TODO
		self.currently_tracked = False
		self.risk_in_row = 0

	def update_position(self, x, y):
		self.x = x
		self.y = y
		self.circle = sg.Point(self.x, self.y).buffer(1)

	def get_outer_poly(self):
		poly = sg.Polygon(self.circle)
		return poly
	