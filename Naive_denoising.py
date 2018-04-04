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
	def __init__(self, dimension_of_observation, noise_variance):
		self.dimension_of_observation = dimension_of_observation
		self.noise_variance = noise_variance

		self.x_ordinate = np.linspace(0, 1, dimension_of_observation)
		self.noise = np.random.normal(0, noise_variance, dimension_of_observation)
		self.observation = Gu(self.x_ordinate) + self.noise
		self.MAP = np.zeros(self.dimension_of_observation)
		self.Lambda = 1

	def Plot_observation(self):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.show()

	def TV_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + self.Lambda * LA.norm(
			np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), 1) / self.dimension_of_observation

	def Gaussian_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + LA.norm(
			np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / (2 * self.dimension_of_observation)

	def TG_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + self.Lambda * LA.norm(
			np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1)) ** 2 + LA.norm(
			np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / (2 * self.dimension_of_observation)

	def Get_MAP(self, method = "TV", Lambda = 1, gamma = 0.1, d = 0.04):
		self.Lambda = Lambda
		if method == "TV":
			input = self.observation
			res = minimize(self.TV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			self.MAP = res.x
			print "Loss:", LA.norm((res.x - self.observation) / sqrt(self.noise_variance))**2, self.Lambda * LA.norm(
			np.delete((res.x),0) - np.delete(res.x, self.dimension_of_observation - 1), 1) / self.dimension_of_observation

		elif method == "Gaussian":
			self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], 
				self.x_ordinate[:, np.newaxis], gamma, d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
			input = self.observation
			res = minimize(self.Gaussian_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			self.MAP = res.x
			print "Loss:" ,LA.norm((res.x - self.observation) / sqrt(self.noise_variance
				))**2 , LA.norm(np.dot(self.inv_sqrt_prior_covariance, res.x)) ** 2

		elif method == "TG":
			self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], 
				self.x_ordinate[:, np.newaxis], gamma, d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
			input = self.observation
			res = minimize(self.TG_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			self.MAP = res.x
			print LA.norm((res.x - self.observation) / sqrt(self.noise_variance))**2, LA.norm(
			np.delete((res.x),0) - np.delete(res.x, self.dimension_of_observation - 1)) ** 2, LA.norm(
			np.dot(self.inv_sqrt_prior_covariance, res.x)) ** 2

		if res.success:
			print("Successfully get MAP from " + method + " prior.\n")
		else:print("Unsuccessfully get MAP.")
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, self.MAP)
		plt.title(method+" prior")
		plt.show()


#############################################################################
if __name__ == '__main__':
	Denoising_example = Naive_denoising(dimension_of_observation = 23, noise_variance = 0.01)
	# Denoising_example.Get_MAP(method = "TV", Lambda = 500)
	# for d in np.linspace(0.04, 0.4, 10):
		# Denoising_example.Get_MAP(method = "Gaussian", d = d)
	# Denoising_example.Get_MAP(method = "Gaussian", d = 0.05)
	# Denoising_example.Get_MAP(method = "Gaussian", d = 0.06)
	# Denoising_example.Get_MAP(method = "Gaussian", d = 0.07)

	Denoising_example.Get_MAP(method = "TG", d = 0.02)
