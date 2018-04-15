import numpy as np
from math import e, sqrt
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import optimize
from scipy.optimize import minimize
import random


# y = Gu(x) + noise
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


	def TV_norm(self, u):
		return self.Lambda * LA.norm(np.delete((u),0) - np.delete(u, 
			self.dimension_of_observation - 1), 1) / self.dimension_of_observation

	def Gaussian_norm(self, u):
		return LA.norm(np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2

	def Phi(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2

	def TV_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + self.Lambda * LA.norm(
			np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), 1) / self.dimension_of_observation

	def Gaussian_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + LA.norm(
			np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2

	def TG_minimizer(self, u):
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 + self.Lambda * LA.norm(
			np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), 1) / self.dimension_of_observation + LA.norm(
			np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2

	def Get_MAP(self, prior = "TV", Lambda = 1, gamma = 0.1, d = 0.04):
		# Maximum a Posterior
		self.Lambda = Lambda
		if prior == "TV":
			input = self.observation
			res = minimize(self.TV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			self.MAP = res.x
			print "Loss:", LA.norm((res.x - self.observation) / sqrt(self.noise_variance
				))**2, self.Lambda * LA.norm(np.delete((res.x),0) - np.delete(res.x, 
				self.dimension_of_observation - 1), 1) / self.dimension_of_observation

		elif prior == "Gaussian":
			self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], 
				self.x_ordinate[:, np.newaxis], gamma, d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
			input = self.observation
			res = minimize(self.Gaussian_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			self.MAP = res.x
			print "Loss:" ,LA.norm((res.x - self.observation) / sqrt(self.noise_variance
				))**2 , LA.norm(np.dot(self.inv_sqrt_prior_covariance, res.x)) ** 2 / 2

		elif prior == "TG":
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
			print("Successfully get MAP from " + prior + " prior.\n")
		else:
			print("Unsuccessfully get MAP.")
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, self.MAP)
		plt.title(method + " prior")
		plt.show()

	def S_pCN(self, Lambda = 1, gamma = 0.1, d = 0.04, sample_size = 1000, splitting_number = 5, beta = 0.1):
		self.Lambda = Lambda
		self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], 
				self.x_ordinate[:, np.newaxis], gamma, d)
		self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

		u_samples = list()
		u_samples.append(self.observation)
		u_mean = np.zeros(self.dimension_of_observation)

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 100) == 0:
				print((i + 0.0) / sample_size)

			u_current = u_samples[-1]
			vj = u_current
			for j in range(splitting_number):
				random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), self.prior_covariance)
				v_prop = sqrt(1 - beta ** 2) * vj + beta * random_walk
				acceptance_rate_R = min(1, e ** (self.TV_norm(vj) - self.TV_norm(v_prop)))
				if random.uniform(0, 1) < acceptance_rate_R:
					vj = v_prop

			acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(vj)))
			if random.uniform(0, 1) < acceptance_rate_Phi:
				u_samples.append(vj)
			else:
				u_samples.append(u_current)

		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, u_mean / sample_size)
		plt.title("S_pCN")
		plt.show()

	def Metropolis_Hastings(self, Lambda = 1, gamma = 0.1, d = 0.04, sample_size = 1000, splitting_number = 5, beta = 0.1):
		self.Lambda = Lambda
		# self.prior_covariance = gaussian_kernel(self.x_ordinate[np.newaxis, :], 
		# 		self.x_ordinate[:, np.newaxis], gamma, d)
		# self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

		u_samples = list()
		u_samples.append(self.observation)
		u_mean = np.zeros(self.dimension_of_observation)

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 100) == 0:
				print((i + 0.0) / sample_size)

			u_current = u_samples[-1]
			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), self.noise_variance * np.eye(self.dimension_of_observation))
			u_prop = u_current + beta * random_walk
			acceptance_rate_I = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
			if random.uniform(0, 1) < acceptance_rate_I:
				u_samples.append(u_prop)
			else:
				u_samples.append(u_current)

		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, u_mean / sample_size)
		plt.title("Metropolis Hastings")
		plt.show()




#############################################################################
if __name__ == '__main__':
	# Denoising_example = Naive_denoising(dimension_of_observation = 23, noise_variance = 0.01)
	# Denoising_example.Get_MAP(prior = "TV", Lambda = 500)
	# Denoising_example.Get_MAP(prior = "TV", Lambda = 400)
	# Denoising_example.Get_MAP(prior = "TV", Lambda = 600)
	# for d in np.linspace(0.04, 0.4, 10):
	# 	Denoising_example.Get_MAP(prior = "Gaussian", d = d)
	# Denoising_example.Get_MAP(prior = "Gaussian", d = 0.05)
	# Denoising_example.Get_MAP(prior = "Gaussian", d = 0.06)
	# Denoising_example.Get_MAP(prior = "Gaussian", d = 0.07)

	Sample_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.04)
	# Sample_example.S_pCN(Lambda = 500, gamma = 0.1, d = 0.02, sample_size = 10000, splitting_number = 5, beta = 0.1)
	Sample_example.Metropolis_Hastings(Lambda = 500, gamma = 0.1, d = 0.02, sample_size = 1000000, splitting_number = 5, beta = 0.01)


