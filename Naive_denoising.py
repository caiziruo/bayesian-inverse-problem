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

# Gaussian kernel is used to generate the covariance matrix
def Gaussian_Kernel(t1, t2, gamma, d):
	return gamma * e ** (- ((t1 - t2) / d)**2 / 2)
	
###########################################################################################################
class Naive_denoising(object):
	def __init__(self, dimension_of_observation, noise_variance, show_observation = False, show_figure = False, save_figure = False):
		self.dimension_of_observation = dimension_of_observation
		# White noise variance.
		self.noise_variance = noise_variance

		# noise and observations are generated by numpy arrays.
		self.x_ordinate = np.linspace(0, 1, dimension_of_observation)
		self.noise = np.random.normal(0, noise_variance, dimension_of_observation)
		self.observation = Gu(self.x_ordinate) + self.noise
		self.MAP = np.zeros(self.dimension_of_observation)
		self.Posterior_Mean = np.zeros(self.dimension_of_observation)

		self.Set_Prior()
		self.show_observation = show_observation
		self.show_figure = show_figure
		self.save_figure = save_figure

	def Set_Prior(self, prior = "TV", Lambda = 1, p = 1, gamma = 0.1, d = 0.04):
		# Lambda: parameter for TV of TG regularization term
		# p: parameter for pV norm
		# gamma and d: parameters of Gaussian kernel
		self.Prior = prior

		if (prior == "TV") or (prior == "phi_Laplace"):
			self.Lambda = Lambda
		elif (prior == "Gaussian"):
			self.gamma = gamma
			self.d = d
			self.prior_covariance = Gaussian_Kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], self.gamma, self.d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
		elif (prior == "TG"):
			self.Lambda = Lambda
			self.gamma = gamma
			self.d = d
			self.prior_covariance = Gaussian_Kernel(self.x_ordinate[np.newaxis, :], self.x_ordinate[:, np.newaxis], self.gamma, self.d)
			self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
		elif (prior == "pV"):
			self.Lambda = Lambda
			self.p = p

	# Definitions of the norms and the potential function.
	def Phi(self, u): 
		return LA.norm((u - self.observation) / sqrt(self.noise_variance))**2 
	def TV_norm(self, u): 
		return self.Lambda * LA.norm(np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), 1)
	def Gaussian_norm(self, u): 
		return LA.norm(np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2
	def pV_norm(self, u):
		return self.Lambda * (self.dimension_of_observation - 1)**(self.p - 1) * LA.norm(np.delete((u),0) - np.delete(u, self.dimension_of_observation - 1), self.p)**self.p

	def TV_minimizer(self, u): return self.Phi(u) + self.TV_norm(u)
	def Gaussian_minimizer(self, u): return self.Phi(u) + self.Gaussian_norm(u)
	def TG_minimizer(self, u): return self.Phi(u) + self.Gaussian_norm(u) + self.TV_norm(u)
	def pV_minimizer(self, u): return self.Phi(u) + self.pV_norm(u)
	def phi_minimizer(self, u): return self.Phi(u) + self.Lambda * abs(2 - self.TV_norm(u))

	# Maximum a Posterior. Computed by optimization package(scipy).
	def Get_MAP(self): 
		input = self.observation
		if self.Prior == "TV":
			res = minimize(self.TV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print("Loss:", self.Phi(res.x), self.TV_norm(res.x))
		elif self.Prior == "Gaussian":
			res = minimize(self.Gaussian_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print("Loss:", self.Phi(res.x), self.Gaussian_norm(res.x))
		elif self.Prior == "TG":
			res = minimize(self.TG_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print("Loss:", self.Phi(res.x), self.TV_norm(res.x), self.Gaussian_norm(res.x))
		elif self.Prior == "pV":
			res = minimize(self.pV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print("Loss:", self.Phi(res.x), self.pV_norm(res.x))
		elif self.Prior == "phi_Laplace":
			res = minimize(self.phi_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
			print("Loss:", self.Phi(res.x), self.phi_minimizer(res.x) - self.Phi(res.x))

		self.MAP = res.x
		if res.success: print("Successfully get MAP from " + self.Prior + " prior.")
		else: print("Unsuccessfully get MAP.")

		self.Plot_MAP()
		return res.x

	def S_pCN(self, sample_size = 1000, splitting_number = 5, beta = 0.1):
		# All the samples are stored in the u_samples list, with each element being a numpy.array.
		u_samples = list()
		# Initialize the first sample u0.
		u_samples.append(self.observation)
		# u_mean is the mean of the samples.
		u_mean = np.zeros(self.dimension_of_observation)
		acc_counter = 0.0

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 10) == 0: print(str((i + 0.0) / sample_size))

			u_current = u_samples[-1]
			vj = u_current
			for j in range(splitting_number):
				random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), self.prior_covariance)
				v_prop = sqrt(1 - beta ** 2) * vj + beta * random_walk 
				acceptance_rate_R = min(1, e ** (self.TV_norm(vj) - self.TV_norm(v_prop)))
				if random.uniform(0, 1) < acceptance_rate_R: vj = v_prop

			acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(vj)))
			if random.uniform(0, 1) < acceptance_rate_Phi: 
				u_samples.append(vj)
				acc_counter += 1
			else: 
				u_samples.append(u_current)

		print("Accept:" + str(acc_counter / sample_size))
		self.Posterior_Mean = u_mean / sample_size
		self.Plot_CM(sample_size, beta)
		return u_samples

	def pCN(self, sample_size = 1000, beta = 0.1):
		u_samples = list()
		u_samples.append(self.observation)
		u_mean = np.zeros(self.dimension_of_observation)
		acc_counter = 0.0

		for i in range(sample_size): 
			u_mean += u_samples[-1]
			if i % (sample_size / 10) == 0: print((i + 0.0) / sample_size)

			u_current = u_samples[-1]
			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), self.prior_covariance)
			u_prop = sqrt(1 - beta ** 2) * u_current + beta * random_walk
			if (self.Prior == "Gaussian"): 
				acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(u_prop)))
			elif (self.Prior == "TG"):
				acceptance_rate_Phi = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
			if random.uniform(0, 1) < acceptance_rate_Phi: 
				u_samples.append(u_prop)
				acc_counter += 1
			else: 
				u_samples.append(u_current)

		print("Accept:" + str(acc_counter / sample_size))
		self.Posterior_Mean = u_mean / sample_size 
		self.Plot_CM(sample_size, beta)
		return u_samples

	def Metropolis_Hastings(self, sample_size = 1000, beta = 0.1):
		u_samples = list()
		u_samples.append(self.observation)
		u_mean = np.zeros(self.dimension_of_observation)
		acc_counter = 0.0

		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 10) == 0: print((i + 0.0) / sample_size)

			if i > 0 and i % (sample_size / 10) == 0:
				self.Posterior_Mean = u_mean / len(u_samples)
				# self.Plot_CM(len(u_samples) - 1, beta)

			u_current = u_samples[-1]
			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_observation), 
				self.noise_variance * np.eye(self.dimension_of_observation))
			u_prop = u_current + beta * random_walk
			if (self.Prior == "TV"): 
				acceptance_rate_I = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
			elif (self.Prior == "Gaussian"):
				acceptance_rate_I = min(1, e ** (self.Gaussian_minimizer(u_current) - self.Gaussian_minimizer(u_prop)))
			elif (self.Prior == "TG"):
				acceptance_rate_I = min(1, e ** (self.TG_minimizer(u_current) - self.TG_minimizer(u_prop)))
			elif (self.Prior == "pV"):
				acceptance_rate_I = min(1, e ** (self.pV_minimizer(u_current) - self.pV_minimizer(u_prop)))
			elif (self.Prior == "phi_Laplace"):
				acceptance_rate_I = min(1, e ** (self.phi_minimizer(u_current) - self.phi_minimizer(u_prop)))
			if random.uniform(0, 1) < acceptance_rate_I: 
				u_samples.append(u_prop)
				acc_counter += 1
			else: 
				u_samples.append(u_current)

		print("Accept:" + str(acc_counter / sample_size))
		self.Posterior_Mean = u_mean / sample_size
		self.Plot_CM(sample_size, beta)
		return u_samples

	def Plot_observation(self):
		plt.plot(self.x_ordinate, self.observation, 'x', label = "observation")
		plt.plot(self.x_ordinate, Gu(self.x_ordinate), 'r', label = "u(t)")
		plt.legend()
		# plt.title("observation")
		if (self.show_figure): plt.show()
		if (self.save_figure): plt.savefig('Observation_'+datetime.now().strftime("_%H%M%S.%f")+'.pdf')
		plt.close('all')

	def Plot_MAP(self):
		if (self.Prior == "TV") or (self.Prior == "phi_Laplace"):
			legend_text = "Lambda = "+str(self.Lambda)
		elif (self.Prior == "Gaussian"):
			legend_text = "gamma = "+str(self.gamma)+"\nd = "+str(self.d)
		elif (self.Prior == "TG"):
			legend_text = "Lambda = "+str(self.Lambda)+"\ngamma = "+str(self.gamma)+"\nd = "+str(self.d)
		elif (self.Prior == "pV"):
			legend_text = "Lambda = "+str(self.Lambda)+"\np = "+str(self.p) 

		if (self.show_observation): plt.plot(self.x_ordinate, self.observation, 'r')
		plt.plot(self.x_ordinate, self.MAP, label = legend_text)
		plt.legend()
		plt.title(self.Prior + " Prior MAP")
		if (self.show_figure): plt.show()
		if (self.save_figure): plt.savefig('MAP_' + self.Prior + datetime.now().strftime("_%H%M%S.%f") + '.pdf')
		# plt.savefig('MAP_' + prior + time.strftime('_%Y%m%d_%H%M%S.%f') + '.pdf')
		plt.close('all')

	def Plot_CM(self, sample_size, beta):
		if (self.Prior == "TV") or (self.Prior == "phi_Laplace"): 
			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)
		elif (self.Prior == "Gaussian"):
			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\ngamma="+str(self.gamma)+"\nd="+str(self.d)
		elif (self.Prior == "TG"):
			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)+"\ngamma="+str(self.gamma)+"\nd="+str(self.d)
		elif (self.Prior == "pV"):
			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)+"\np="+str(self.p)

		if (self.show_observation): plt.plot(self.x_ordinate, self.observation, 'r')
		plt.plot(self.x_ordinate, self.Posterior_Mean, label = legend_text)
		plt.legend()
		plt.title(self.Prior + " Prior Posterior Mean")
		if (self.show_figure): plt.show()
		if (self.save_figure): plt.savefig('CM_'+ self.Prior +datetime.now().strftime("_%H%M%S.%f")+'.pdf')
		plt.close('all')


###########################################################################################################
if __name__ == '__main__':
	"""
	Maximum A Posterior Experiments with different priors.
	"""
	Denoising_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.02, show_observation = True, show_figure = True, save_figure = False)
	# Denoising_example.Plot_observation()
	# Denoising_example.Set_Prior(prior = "pV", Lambda = 0.4, p = 2)
	# Denoising_example.Get_MAP()
	# Denoising_example.Metropolis_Hastings(sample_size = 100000, beta = 0.3)
	# Denoising_example.Set_Prior(prior = "pV", Lambda = 2, p = 1.5)
	# Denoising_example.Get_MAP()
	# Denoising_example.Metropolis_Hastings(sample_size = 100000, beta = 0.3)
	# Denoising_example.Set_Prior(prior = "pV", Lambda = 10, p = 1)
	# Denoising_example.Get_MAP()
	# Denoising_example.Metropolis_Hastings(sample_size = 100000, beta = 0.2)


	# Denoising_example.Set_Prior(prior = "TV", Lambda = 50)
	# Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "Gaussian", gamma = 0.1, d = 0.04)
	# Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.04)
	# Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.5)
	# Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "TV", Lambda = 40)
	# Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "TV", Lambda = 60)
	# Denoising_example.Get_MAP()



	# for d in np.linspace(0.04, 0.4, 10):
	# 	Denoising_example.Set_Prior(prior = "Gaussian", gamma = 0.1, d = d)
	# 	Denoising_example.Get_MAP()
	# Denoising_example.Set_Prior(prior = "phi_Laplace", Lambda = 10**8)
	# Denoising_example.Get_MAP()

	"""
	MCMC
	"""
	# Sample_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.001, show_observation = False, show_figure = True, save_figure = False)
	# Sample_example.Set_Prior("TG", Lambda = 50, gamma = 1, d = 0.04)
	# Sample_example.S_pCN(sample_size = 1000, splitting_number = 5, beta = 0.1)
	# Sample_example.Set_Prior("TV", Lambda = 50)
	# Sample_example.Metropolis_Hastings(sample_size = 1000, beta = 0.01)


	Sample_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.01, show_observation = False, show_figure = False, save_figure = False)
	# Sample_example.Set_Prior(prior = "phi_Laplace", Lambda = 1000)
	# Sample_example.Metropolis_Hastings(sample_size = 1000000, beta = 0.0005)
	Sample_example.Set_Prior(prior = "TV", Lambda = 1000)
	Sample_example.Metropolis_Hastings(sample_size = 10000, beta = 0.002)
	# Sample_example.Set_Prior(prior = "Gaussian", gamma = 0.1, d = 0.2)
	# Sample_example.pCN(sample_size = 3000000, beta = 0.05)
	# Sample_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.04)
	# Sample_example.pCN(sample_size = 2000000, beta = 0.038)
	# Sample_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.03)
	# Sample_example.pCN(sample_size = 2000000, beta = 0.035)
	# Sample_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.02)
	# Sample_example.pCN(sample_size = 2000000, beta = 0.025)
	# Sample_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.01)
	# Sample_example.pCN(sample_size = 2000000, beta = 0.015)
	# Sample_example.Set_Prior(prior = "TG", Lambda = 50, gamma = 0.1, d = 0.005)
	# Sample_example.pCN(sample_size = 2000000, beta = 0.01)


	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000000, beta = 0.07)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 60, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000000, beta = 0.06)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 70, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000000, beta = 0.05)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 80, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000000, beta = 0.04)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 90, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000000, beta = 0.03)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 100, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 5000, beta = 0.03)


	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.9)
	# Sample_example.Metropolis_Hastings(sample_size = 10000, beta = 0.035)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.8)
	# Sample_example.Metropolis_Hastings(sample_size = 10000, beta = 0.04)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.7)
	# Sample_example.Metropolis_Hastings(sample_size = 10000, beta = 0.045)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.6)
	# Sample_example.Metropolis_Hastings(sample_size = 10000, beta = 0.055)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 50, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 100000, beta = 0.07)


	"""
	prior = pV, Posterior Mean with different dimensions.
	"""
	# Sample_example = Naive_denoising(dimension_of_observation = 89, noise_variance = 0.01, show_observation = True, show_figure = False, save_figure = True)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 150, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 1000000, beta = 0.015)
	# Sample_example = Naive_denoising(dimension_of_observation = 177, noise_variance = 0.01, show_observation = True, show_figure = False, save_figure = True)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 220, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 2000000, beta = 0.009)
	# Sample_example = Naive_denoising(dimension_of_observation = 353, noise_variance = 0.01, show_observation = True, show_figure = False, save_figure = True)
	# Sample_example.Set_Prior(prior = "pV", Lambda = 500, p = 0.5)
	# Sample_example.Metropolis_Hastings(sample_size = 8000000, beta = 0.003)

	
	