import matplotlib.pyplot as plt
import numpy as np

def Gu(x):
	return ((x >= 1 / 3.0) & (x < 2 / 3.0)) * 1

class Observation(object):
	"""the observation object describes the formula y = G(u) + noise"""
	
	def __init__(self, dimension_of_observation):
		self.x_ordinate = np.linspace(0, 1, dimension_of_observation)
		# self.noise_mean = np.zeros(dimension_of_observation)
		self.noise = np.random.normal(0, 0.02**2, dimension_of_observation)
		"""white noise"""
		self.observation = Gu(self.x_ordinate) + self.noise

	def plot_observation(self):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.show()



