import numpy as np
from math import e, sqrt
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import optimize
from scipy.optimize import minimize
import random
import time
from datetime import datetime

# y = Gu(x) + white noise
def Gu(x):
	return ((x >= 1 / 3.0) & (x < 2 / 3.0)) * 1

def Gaussian_Kernel(t1, t2, gamma, d):
	return gamma * e ** (- ((t1 - t2) / d)**2 / 2)
	
###########################################################################################################
class Naive_denoising(object):
	def __init__(self, dimension_of_observation, noise_variance):
		self.dimension_of_observation = dimension_of_observation
		# White noise variance.
		self.noise_variance = noise_variance

		# noise and observations are numpy arrays.
		self.x_ordinate = np.linspace(0, 1, dimension_of_observation)
		self.noise = np.random.normal(0, noise_variance, dimension_of_observation)
		self.observation = Gu(self.x_ordinate) + self.noise
		self.MAP = np.zeros(self.dimension_of_observation)
		self.Posterior_Mean = np.zeros(self.dimension_of_observation)
		# Parameter in the TV regularization term.
		self.Lambda = 1

	# Definitions of the norms and the potential function.
	def TV_norm(self, u): 
		return self.Lambda * LA.norm(np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), 1) / self.dimension_of_observation
	def Gaussian_norm(self, u): 
		return LA.norm(np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2
	def Phi(self, u): 
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 
	def TV_minimizer(self, u): return self.Phi(u) + self.TV_norm(u)
	def Gaussian_minimizer(self, u): return self.Phi(u) + self.Gaussian_norm(u)
	def TG_minimizer(self, u): return self.Phi(u) + self.Gaussian_norm(u) + self.TV_norm(u)

	# Maximum a Posterior. Computed by optimization package(scipy).
	# Gamma and d are parameters of the Gaussian Kernel.
	def Get_MAP(self, prior = "TV", Lambda = 1, gamma = 0.1, d = 0.04):
		self.Lambda = Lambda

		input = self.observation
		if prior == "TV":
			res = minimize(self.TV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print "Loss:", self.Phi(res.x), self.TV_norm(res.x)
		elif prior == "Gaussian":
			# Define the prior covariance and precision matrix.
			self.prior_covariance = Gaussian_Kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], gamma, d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

			res = minimize(self.Gaussian_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print "Loss:", self.Phi(res.x), self.Gaussian_norm(res.x)
		elif prior == "TG":
			self.prior_covariance = Gaussian_Kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], gamma, d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

			res = minimize(self.TG_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print "Loss:", self.Phi(res.x), self.TV_norm(res.x), self.Gaussian_norm(res.x)

		self.MAP = res.x
		if res.success: print("Successfully get MAP from " + prior + " prior.\n")
		else: print("Unsuccessfully get MAP.")

		self.Plot_MAP(prior, Lambda, gamma, d)

	def S_pCN(self, prior = "TG", Lambda = 1, gamma = 0.1, d = 0.04, sample_size = 1000, splitting_number = 5, beta = 0.1):
		self.Lambda = Lambda
		self.prior_covariance = Gaussian_Kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], gamma, d)
		self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

		# All the samples are stored in the u_samples list, with each element being a numpy.array.
		u_samples = list()
		# Initialize the first sample u0.
		u_samples.append(self.observation)
		# u_mean is the mean of the samples.
		u_mean = np.zeros(self.dimension_of_observation)

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 100) == 0: print((i + 0.0) / sample_size)

			u_current = u_samples[-1]
			vj = u_current
			for j in range(splitting_number):
				random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), self.prior_covariance)
				v_prop = sqrt(1 - beta ** 2) * vj + beta * random_walk
				acceptance_rate_R = min(1, e ** (self.TV_norm(vj) - self.TV_norm(v_prop)))
				if random.uniform(0, 1) < acceptance_rate_R: vj = v_prop

			acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(vj)))
			if random.uniform(0, 1) < acceptance_rate_Phi: u_samples.append(vj)
			else: u_samples.append(u_current)

		self.Posterior_Mean = u_mean / sample_size
		self.Plot_CM(prior, Lambda, gamma, d, sample_size, splitting_number, beta)
		return u_samples

	def Metropolis_Hastings(self, prior = "TV", Lambda = 1, gamma = 0.1, d = 0.04, sample_size = 1000, splitting_number = 5, beta = 0.1):
		self.Lambda = Lambda

		u_samples = list()
		u_samples.append(self.observation)
		u_mean = np.zeros(self.dimension_of_observation)

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 100) == 0: print((i + 0.0) / sample_size)

			u_current = u_samples[-1]
			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), 
				self.noise_variance * np.eye(self.dimension_of_observation))
			u_prop = u_current + beta * random_walk
			acceptance_rate_I = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
			if random.uniform(0, 1) < acceptance_rate_I: u_samples.append(u_prop)
			else: u_samples.append(u_current)

		self.Posterior_Mean = u_mean / sample_size
		self.Plot_CM(prior, Lambda, gamma, d, sample_size, splitting_number, beta)
		return u_samples

	def Plot_observation(self):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.show()
		plt.close('all')

	def Plot_MAP(self, prior, Lambda, gamma, d):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, self.MAP)
		if (prior == "TV"):
			plt.text(0.8, 0.9, "Lambda = " + str(Lambda))
		elif (prior == "Gaussian"):
			plt.text(0.8, 0.9, "gamma = " + str(gamma))
			plt.text(0.8, 0.8, "d = " + str(d))
		plt.title(prior + " Prior MAP")
		# plt.show()
		plt.savefig('MAP_'+ prior +datetime.now().strftime("_%H%M%S.%f")+'.pdf')
		# plt.savefig('MAP_'+ prior +time.strftime('_%Y%m%d_%H%M%S.%f')+'.pdf')
		plt.close('all')

	def Plot_CM(self, prior, Lambda, gamma, d, sample_size, splitting_number, beta):
		plt.plot(self.x_ordinate, self.observation, 'x')
		plt.plot(self.x_ordinate, self.Posterior_Mean)
		plt.text(0.8, 0.9, "Sample size = " + str(sample_size))
		plt.text(0.8, 0.8, "beta = " + str(beta))

		if (prior == "TV"): 
			plt.text(0.8, 0.7, "Lambda = " + str(Lambda))
		elif (prior == "Gaussian"): 
			plt.text(0.8, 0.7, "d = " + str(d))
		elif (prior == "TG"):
			plt.text(0.8, 0.7, "Lambda = " + str(Lambda))
			plt.text(0.8, 0.6, "d = " + str(d))

		plt.title(prior + " Prior Posterior Mean")
		# plt.show()
		plt.savefig('CM_'+ prior +datetime.now().strftime("_%H%M%S.%f")+'.pdf')
		plt.close('all')


###########################################################################################################
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

	Sample_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.001)
	Sample_example.S_pCN(prior = "TG", Lambda = 500, gamma = 0.1, d = 0.02, sample_size = 10000, splitting_number = 5, beta = 0.1)
	Sample_example.Metropolis_Hastings(Lambda = 500, prior = "TV", gamma = 0.1, d = 0.02, sample_size = 10000, splitting_number = 5, beta = 0.01)
	Sample_example = Naive_denoising(dimension_of_observation = 177, noise_variance = 0.001)
	Sample_example.Metropolis_Hastings(Lambda = 500, prior = "TV", gamma = 0.1, d = 0.02, sample_size = 10000, splitting_number = 5, beta = 0.01)
	Sample_example = Naive_denoising(dimension_of_observation = 353, noise_variance = 0.001)
	Sample_example.Metropolis_Hastings(Lambda = 500, prior = "TV", gamma = 0.1, d = 0.02, sample_size = 10000, splitting_number = 5, beta = 0.01)