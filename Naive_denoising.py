import numpy as np
from math import e, sqrt
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import optimize
from scipy.optimize import minimize

def Gu(x):
	return ((x >= 1 / 3.0) & (x < 2 / 3.0)) * 1

def gaussian_kernel(t1, t2, gamma, d):
	return gamma * e ** (- ((t1 - t2) / d)**2 / 2)
	
#############################################################################
class Naive_denoising(object):
	
	def __init__(self, dimension_of_observation, noise_variance, Lambda):
		self.gamma = 0.1
		self.d = 0.4
		self.noise_variance = noise_variance
		self.Lambda = Lambda
		self.dimension_of_observation = dimension_of_observation

		self.x_ordinate = np.linspace(0, 1, dimension_of_observation)
		self.noise = np.random.normal(0, noise_variance, dimension_of_observation)
		self.observation = Gu(self.x_ordinate) + self.noise
		self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], self.gamma, self.d)


	def Plot_observation(self):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.show()

	def TV_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + self.Lambda * LA.norm(np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1)) ** 2

	def Minimize(self, method = "TV"):
		if method == "TV":
			input = np.array(Gu(self.x_ordinate)*0)
			res = minimize(self.TV_minimizer, input, method = 'CG', tol = 1e-6)
			# print(res)

			plt.plot(self.x_ordinate, self.observation, 'x')
			plt.plot(self.x_ordinate, res.x)
			plt.show()

		elif method == "Gaussian":
			pass
		elif method == "TG":
			pass



if __name__ == '__main__':
	dimension_of_observation = 23
	noise_variance = 0.01
	Lambda = 10
	Denoising_example = Naive_denoising(dimension_of_observation, noise_variance, Lambda)
	Denoising_example.Minimize("TV")