import numpy as np
from math import e, sqrt
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import optimize
from scipy.optimize import minimize
import random
import time
from datetime import datetime
import sys
import cv2

# y = Gu(x) + white noise
# def Gu_initialize(x):
# 	return ((x >= 1 / 3.0) & (x <= 2 / 3.0)) * 1

# Gaussian kernel is used to generate the covariance matrix
# def Gaussian_Kernel(t1, t2, gamma, d):
# 	return gamma * e ** (- ((t1 - t2) / d)**2 / 2)
	
###########################################################################################################
class Motion_Deblur(object):
	def __init__(self, noise_variance, show_observation = False, show_figure = False, save_figure = False, load_observation = "", load_figure = ""):
		img = cv2.imread(load_figure, 0) # load grayscale image

		# if (dimension_of_unknown - 1) % (dimension_of_observation - 1) != 0:
		# # check if the dimensions are matched.
		# 	print("Dimensions of observation and unknown are not matched.")
		# 	sys.exit(0)

		# self.coarse_factor is the compression ratio from unknown to observation.
		# self.coarse_factor = int((dimension_of_unknown - 1) / (dimension_of_observation - 1))
		self.dimension_ob_row = img.shape[0]
		self.dimension_ob_column = img.shape[1]
		self.dimension_of_observation = self.dimension_ob_row * self.dimension_ob_column
		self.dimension_un_row = self.dimension_ob_row
		self.dimension_un_column = self.dimension_ob_column
		self.dimension_of_unknown = self.dimension_un_row * self.dimension_un_column

		self.kernel_size = 10
		self.kernel_motion_blur = np.zeros((self.kernel_size, self.kernel_size))
		self.kernel_motion_blur[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
		self.kernel_motion_blur = self.kernel_motion_blur / self.kernel_size
		self.noise_variance = noise_variance
		if (load_observation == ""): # Motion blur and add noise
			self.kernel_size = 10
			self.kernel_motion_blur = np.zeros((self.kernel_size, self.kernel_size))
			self.kernel_motion_blur[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
			self.kernel_motion_blur = self.kernel_motion_blur / self.kernel_size
			self.observation = cv2.filter2D(img, -1, self.kernel_motion_blur)

			self.noise = np.random.normal(0, self.noise_variance, self.dimension_of_observation)
			self.noise = np.reshape(self.noise, (self.dimension_ob_row, self.dimension_ob_column))
			self.observation = self.observation + self.noise
		else:
			self.observation = cv2.imread(load_observation, 0)

		self.unknown = self.observation

		# initialize MAP and posterior mean
		# self.MAP = np.zeros(self.dimension_of_unknown)
		self.Posterior_Mean = np.zeros((self.dimension_ob_row, self.dimension_ob_column))

		# self.Set_Prior()
		self.show_observation = show_observation
		self.show_figure = show_figure
		self.save_figure = save_figure


	def Gu(self, u): # an operator mapping from unknown to observation
		# Gu = np.linspace(0, 1, self.dimension_of_observation)
		# for i in range(self.dimension_of_observation):
		# 	Gu[i] = u[i * self.coarse_factor]
		# return np.dot(self.operator_matrix, u)
		return cv2.filter2D(self.unknown, -1, self.kernel_motion_blur)

	def Set_Prior(self, prior = "TV", Lambda = 1, p = 1, gamma = 0.1, d = 0.04):
	# 	# Lambda: parameter for TV of TG regularization term
	# 	# p: parameter for pV norm
	# 	# gamma and d: parameters of Gaussian kernel
		self.Prior = prior

		if (prior == "TV") or (prior == "phi_Laplace"):
			self.Lambda = Lambda
			self.p = 0
			self.gamma = 0
			self.d = 0
	# 	elif (prior == "pV"):
	# 		self.Lambda = Lambda
	# 		self.p = p
	# 		self.gamma = 0
	# 		self.d = 0
	# 	elif (prior == "Gaussian"):
	# 		self.gamma = gamma
	# 		self.d = d
	# 		self.Lambda = 0
	# 		self.p = 0
	# 		self.prior_covariance = Gaussian_Kernel(self.unknown_ordinate[np.newaxis, :], self.unknown_ordinate[:, np.newaxis], self.gamma, self.d)
	# 		self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
	# 	elif (prior == "TG") or (prior == "phi_G"):
	# 		self.Lambda = Lambda
	# 		self.gamma = gamma
	# 		self.d = d
	# 		self.p = 0
	# 		self.prior_covariance = Gaussian_Kernel(self.unknown_ordinate[np.newaxis, :], self.unknown_ordinate[:, np.newaxis], self.gamma, self.d)
	# 		self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))
	# 	elif (prior == "pG"):
	# 		self.Lambda = Lambda
	# 		self.p = p
	# 		self.gamma = gamma
	# 		self.d = d
	# 		self.prior_covariance = Gaussian_Kernel(self.unknown_ordinate[np.newaxis, :], self.unknown_ordinate[:, np.newaxis], self.gamma, self.d)
	# 		self.inv_sqrt_prior_covariance = LA.inv(np.sqrt(self.prior_covariance))

	# Definitions of the norms and the potential function.
	def Phi(self, u): return LA.norm((self.Gu(u) - self.observation)/(sqrt(self.noise_variance) * 255))**2 / 2

	def TV_norm(self, u): 
		row_variation_norm = np.sum(abs( np.delete(u, 0, 1) - np.delete(u, self.dimension_ob_column - 1, 1) )) / (255 ** 2)
		column_variation_norm = np.sum(abs( np.delete(u, 0, 0) - np.delete(u, self.dimension_ob_row - 1, 0) )) / (255 ** 2)

		return self.Lambda * (row_variation_norm + column_variation_norm)

	# def Gaussian_norm(self, u): return LA.norm(np.dot(self.inv_sqrt_prior_covariance, u)) ** 2 / 2
	# def pV_norm(self, u): return self.Lambda * (self.dimension_of_unknown - 1)**(self.p - 1) * LA.norm(np.delete((u),0) - np.delete(u, self.dimension_of_unknown - 1), self.p)**self.p

	def TV_minimizer(self, u): 
		# print(self.Phi(u), self.TV_norm(u))
		return self.Phi(u) + self.TV_norm(u)
	# def Gaussian_minimizer(self, u): return self.Phi(u) + self.Gaussian_norm(u)
	# def TG_minimizer(self, u): return self.Phi(u) + self.TV_norm(u) + self.Gaussian_norm(u)
	# def pV_minimizer(self, u): return self.Phi(u) + self.pV_norm(u)
	# def pG_minimizer(self, u): return self.Phi(u) + self.pV_norm(u) + self.Gaussian_norm(u)
	# def phi_minimizer(self, u): return self.Phi(u) + self.Lambda * abs(2 - self.TV_norm(u))
	# def phi_G_minimizer(self, u): return self.Phi(u) + self.Lambda * abs(2 - self.TV_norm(u)) + self.Gaussian_norm(u)

	# Maximum a Posterior. Computed by optimization package(scipy).
# 	def Get_MAP(self, save_npy = False): 
# 		input = self.unknown
# 		if self.Prior == "TV":
# 			res = minimize(self.TV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
# 			print("Loss:", self.Phi(res.x), self.TV_norm(res.x))
# 		elif self.Prior == "Gaussian":
# 			res = minimize(self.Gaussian_minimizer, input, method = 'TNC', tol = 1e-6)
# 			print("Loss:", self.Phi(res.x), self.Gaussian_norm(res.x))
# 		elif self.Prior == "TG":
# 			res = minimize(self.TG_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
# 			print("Loss:", self.Phi(res.x), self.TV_norm(res.x), self.Gaussian_norm(res.x))
# 		elif self.Prior == "pV":
# 			res = minimize(self.pV_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
# 			print("Loss:", self.Phi(res.x), self.pV_norm(res.x))
# 		elif self.Prior == "phi_Laplace":
# 			res = minimize(self.phi_minimizer, input, method = 'L-BFGS-B', tol = 1e-6)
# 			print("Loss:", self.Phi(res.x), self.phi_minimizer(res.x) - self.Phi(res.x))

# 		self.MAP = res.x
# 		if res.success: print("Successfully get MAP from " + self.Prior + " prior.")
# 		else: print("Unsuccessfully get MAP.")

# 		self.Plot_MAP()

# 		if (res.success and save_npy):
# 			samples_np = np.array(self.MAP)
# 			np.save("MAP" + self.Prior + "_time_"+ datetime.now().strftime("%H%M%S.%f") + ".npy", samples_np)
# 			# np.save("Observation_time_" + datetime.now().strftime("%H%M%S.%f") + ".npy", self.observation)
# 			text_file = open("MAP" + self.Prior + "_time_"+ datetime.now().strftime("%H%M%S.%f") + ".txt", "w")
# 			text_file.write("Lambda = " + str(self.Lambda) + "\np = " + str(self.p) + "\ngamma = " + str(self.gamma) + "\nd = " + str(self.d) )

# 		return res.x

# 	# def S_pCN(self, sample_size = 1000, splitting_number = 5, beta = 0.1):
# 	# 	# All the samples are stored in the u_samples list, with each element being a numpy.array.
# 	# 	u_samples = list()
# 	# 	# Initialize the first sample u0.
# 	# 	u_samples.append(self.unknown)
# 	# 	# u_mean is the mean of the samples.
# 	# 	u_mean = np.zeros(self.dimension_of_unknown)
# 	# 	acc_counter = 0.0

# 	# 	for i in range(sample_size):
# 	# 		u_mean += u_samples[-1]
# 	# 		if i % (sample_size / 10) == 0: print(str((i + 0.0) / sample_size))

# 	# 		u_current = u_samples[-1]
# 	# 		vj = u_current
# 	# 		for j in range(splitting_number):
# 	# 			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_unknown), self.prior_covariance)
# 	# 			v_prop = sqrt(1 - beta ** 2) * vj + beta * random_walk 
# 	# 			acceptance_rate_R = min(1, e ** (self.TV_norm(vj) - self.TV_norm(v_prop)))
# 	# 			if random.uniform(0, 1) < acceptance_rate_R: vj = v_prop

# 	# 		acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(vj)))
# 	# 		if random.uniform(0, 1) < acceptance_rate_Phi: 
# 	# 			u_samples.append(vj)
# 	# 			acc_counter += 1
# 	# 		else: 
# 	# 			u_samples.append(u_current)

# 	# 	print("Accept:" + str(acc_counter / sample_size))
# 	# 	self.Posterior_Mean = u_mean / sample_size
# 	# 	self.Plot_CM(sample_size, beta)
# 	# 	return u_samples

# 	def pCN(self, sample_size = 1000, beta = 0.1, save_npy = False, initialize_unknown_from_npy = ""):
# 		u_samples = list()
# 		if (initialize_unknown_from_npy != ""): 
# 			print("Initializing unknown from " + initialize_unknown_from_npy + "...")
# 			npy_end_sample = np.load(initialize_unknown_from_npy) 
# 			self.unknown = npy_end_sample[-1].copy()
# 			# npy_end_sample.close()
# 			del npy_end_sample

# 		u_samples.append(self.unknown) 
# 		u_mean = np.zeros(self.dimension_of_unknown)
# 		acc_counter = 0.0

# 		print("pCN: ")
# 		for i in range(sample_size): 
# 			u_mean += u_samples[-1]
# 			if i % (sample_size / 10) == 0: print((i + 0.0) / sample_size)

# 			u_current = u_samples[-1]
# 			random_walk = np.random.multivariate_normal(np.zeros(self.dimension_of_unknown), self.prior_covariance)
# 			u_prop = sqrt(1 - beta ** 2) * u_current + beta * random_walk
# 			if (self.Prior == "Gaussian"): 
# 				acceptance_rate_Phi = min(1, e ** (self.Phi(u_current) - self.Phi(u_prop)))
# 			elif (self.Prior == "TG"):
# 				acceptance_rate_Phi = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
# 			elif (self.Prior == "pG"):
# 				acceptance_rate_Phi = min(1, e ** (self.pV_minimizer(u_current) - self.pV_minimizer(u_prop)))
# 			elif (self.Prior == "phi_G"):
# 				acceptance_rate_Phi = min(1, e ** (self.phi_minimizer(u_current) - self.phi_minimizer(u_prop)))
# 			if random.uniform(0, 1) < acceptance_rate_Phi: 
# 				u_samples.append(u_prop)
# 				acc_counter += 1
# 			else: 
# 				u_samples.append(u_current)

# 		print("Accept:" + str(acc_counter / sample_size))
# 		self.Posterior_Mean = u_mean / sample_size 
# 		self.Plot_CM(sample_size, beta)

# 		# save the samples as .npy files and the parameters are recorded in .txt files.
# 		if (save_npy): 
# 			samples_np = np.array(u_samples) 
# 			np.save(self.Prior + "_time_"+ datetime.now().strftime("%H%M%S.%f") + ".npy", samples_np)
# 			np.save("Observation_time_" + datetime.now().strftime("%H%M%S.%f") + ".npy", self.observation)
# 			text_file = open(self.Prior + "_time_"+ datetime.now().strftime("%H%M%S.%f") + ".txt", "w")
# 			text_file.write("Lambda = " + str(self.Lambda) + "\np = " + str(self.p) + "\ngamma = " + str(self.gamma) + "\nd = " + str(self.d)+ "\nsample size = " + str(sample_size) + "\nsample step = " + str(beta) + "\nAccept:" + str(acc_counter / sample_size))

# 		return u_samples

	def Metropolis_Hastings(self, sample_size = 1000, beta = 0.1, save_npy = False, initialize_unknown_from_npy = ""):
		if (initialize_unknown_from_npy != ""):
			print("Initializing unknown from " + initialize_unknown_from_npy + "...")
			npy_end_sample = np.load(initialize_unknown_from_npy)
			self.unknown = npy_end_sample[-1].copy()
			del npy_end_sample

		u_samples = list()
		u_samples.append(self.unknown)
		u_mean = np.zeros((self.dimension_ob_row, self.dimension_ob_column))
		acc_counter = 0.0

		print("Metropolis Hastings:")
		for i in range(sample_size):
			u_mean += u_samples[-1]
			if i % (sample_size / 10) == 0: print((i + 0.0) / sample_size)

			# if i > 0 and i % (sample_size / 10) == 0: 
			# 	self.Posterior_Mean = u_mean / len(u_samples) 
				# self.Plot_CM(len(u_samples) - 1, beta)

			u_current = u_samples[-1]
			random_walk = np.random.normal(0, self.noise_variance, self.dimension_of_unknown)
			random_walk = np.reshape(random_walk, (self.dimension_ob_row, self.dimension_ob_column))
			u_prop = u_current + beta * random_walk
			if (self.Prior == "TV"):
				acceptance_rate_I = min(1, e ** (self.TV_minimizer(u_current) - self.TV_minimizer(u_prop)))
			# elif (self.Prior == "Gaussian"):
			# 	acceptance_rate_I = min(1, e ** (self.Gaussian_minimizer(u_current) - self.Gaussian_minimizer(u_prop)))
			# elif (self.Prior == "TG"):
			# 	acceptance_rate_I = min(1, e ** (self.TG_minimizer(u_current) - self.TG_minimizer(u_prop)))
			# elif (self.Prior == "pV"):
			# 	acceptance_rate_I = min(1, e ** (self.pV_minimizer(u_current) - self.pV_minimizer(u_prop)))
			# elif (self.Prior == "phi_Laplace"):
			# 	acceptance_rate_I = min(1, e ** (self.phi_minimizer(u_current) - self.phi_minimizer(u_prop)))
			if random.uniform(0, 1) < acceptance_rate_I:
				u_samples.append(u_prop)
				acc_counter += 1
			else:
				u_samples.append(u_current)


		print("Accept:" + str(acc_counter / sample_size))
		self.Posterior_Mean = u_mean / sample_size
		# self.Plot_CM(sample_size, beta)
		cv2.imwrite("posterior_mean.jpg", self.Posterior_Mean)

		# save the samples as .npy files and the parameters are recorded in .txt files.
		if (save_npy): 
			samples_np = np.array(u_samples) 
			np.save("TV_Motion_Deblur_Samples.npy", samples_np)
			# np.save("Observation_time_" + datetime.now().strftime("%H%M%S.%f") + ".npy", self.observation)
			# text_file = open(self.Prior + "_time_"+ datetime.now().strftime("%H%M%S.%f") + ".txt", "w")
			# text_file.write("Lambda = " + str(self.Lambda) + "\np = " + str(self.p) + "\ngamma = " + str(self.gamma) + "\nd = " + str(self.d)+ "\nsample size = " + str(sample_size) + "\nsample step = " + str(beta) + "\nAccept:" + str(acc_counter / sample_size))

		return u_samples

# 	def Plot_observation(self):
# 		plt.plot(self.observation_ordinate, self.observation, 'x', label = "observation")
# 		plt.plot(self.observation_ordinate, Gu_initialize(self.observation_ordinate), 'r', label = "u(t)")
# 		plt.legend()
# 		plt.title("observation")
# 		if (self.show_figure): plt.show()
# 		if (self.save_figure): plt.savefig('Observation_'+datetime.now().strftime("_%H%M%S.%f")+'.pdf')
# 		plt.close('all')

# 	def Plot_MAP(self):
# 		if (self.Prior == "TV") or (self.Prior == "phi_Laplace"):
# 			legend_text = "Lambda = "+str(self.Lambda)
# 		elif (self.Prior == "Gaussian"):
# 			legend_text = "gamma = "+str(self.gamma)+"\nd = "+str(self.d)
# 		elif (self.Prior == "TG"):
# 			legend_text = "Lambda = "+str(self.Lambda)+"\ngamma = "+str(self.gamma)+"\nd = "+str(self.d)
# 		elif (self.Prior == "pV"):
# 			legend_text = "Lambda = "+str(self.Lambda)+"\np = "+str(self.p) 

# 		if (self.show_observation): plt.plot(self.observation_ordinate, self.observation, 'x')
# 		plt.plot(self.unknown_ordinate, self.MAP, label = legend_text)
# 		plt.legend()
# 		plt.title(self.Prior + " Prior MAP")
# 		if (self.show_figure): plt.show()
# 		if (self.save_figure): plt.savefig('MAP_' + self.Prior + datetime.now().strftime("_%H%M%S.%f") + '.pdf')
# 		# plt.savefig('MAP_' + prior + time.strftime('_%Y%m%d_%H%M%S.%f') + '.pdf')
# 		plt.close('all')

# 	def Plot_CM(self, sample_size, beta):
# 		legend_text = "123"
# 		if (self.Prior == "TV") or (self.Prior == "phi_Laplace"):
# 			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)
# 		elif (self.Prior == "Gaussian"):
# 			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\ngamma="+str(self.gamma)+"\nd="+str(self.d)
# 		elif (self.Prior == "TG"):
# 			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)+"\ngamma="+str(self.gamma)+"\nd="+str(self.d)
# 		elif (self.Prior == "pV"):
# 			legend_text = "Sp="+str(sample_size)+"\nbeta="+str(beta)+"\nLambda="+str(self.Lambda)+"\np="+str(self.p)

# 		if (self.show_observation): plt.plot(self.observation_ordinate, self.observation, 'x')
# 		plt.plot(self.unknown_ordinate, self.Posterior_Mean, label = legend_text)
# 		plt.legend()
# 		plt.title(self.Prior + " Prior Posterior Mean")
# 		if (self.show_figure): plt.show()
# 		if (self.save_figure): plt.savefig('CM_'+ self.Prior +datetime.now().strftime("_%H%M%S.%f")+'.pdf')
# 		plt.close('all')

# def Plot_CM_from_npy(npy_name, Observation_npy, Prior, Lambda, p, gamma, d, show_observation = False, show_figure = False, save_figure = False):
# 	samples_np = np.load(npy_name)
# 	dimension_of_unknown = samples_np.shape[1]
# 	unknown_ordinate = np.linspace(0, 1, dimension_of_unknown)

# 	npy_mean = np.mean(samples_np, axis=0)
# 	sample_size = samples_np.shape[0] - 1


# 	if (Prior == "TV") or (Prior == "phi_Laplace"): legend_text = "Lambda="+str(Lambda)
# 	elif (Prior == "Gaussian"): legend_text = "gamma="+str(gamma)+"\nd="+str(d)
# 	elif (Prior == "TG"): legend_text = "Lambda="+str(Lambda)+"\ngamma="+str(gamma)+"\nd="+str(d)
# 	elif (Prior == "pV"): legend_text = "Lambda="+str(Lambda)+"\np="+str(p)

# 	if (show_observation): 
# 		observation = np.load(Observation_npy)
# 		dimension_of_observation = observation.shape[0]
# 		observation_ordinate = np.linspace(0, 1, dimension_of_observation)
# 		plt.plot(observation_ordinate, observation, 'x', label = "observation")
# 	plt.plot(unknown_ordinate, npy_mean, label = legend_text)
# 	plt.legend()
# 	plt.title(Prior + " Prior Posterior Mean")
# 	if (save_figure): plt.savefig('CM_'+ Prior +datetime.now().strftime("_%H%M%S.%f")+'.pdf')
# 	if (show_figure): plt.show()
# 	plt.close('all')


###########################################################################################################
if __name__ == '__main__':
	Sample_example = Motion_Deblur(noise_variance = 10.0, show_observation = False, show_figure = False, save_figure = False, load_figure = "figure.jpg", load_observation = "figure3.jpg")
	Sample_example.Set_Prior(prior = "TV", Lambda = 200.0)

	for i in range(100):
		Sample_example.Metropolis_Hastings(sample_size = 5000, beta = 0.05, save_npy = True, initialize_unknown_from_npy = "TV_Motion_Deblur_Samples.npy")

	# Sample_example.Metropolis_Hastings(sample_size = 5000, beta = 0.01, save_npy = True)

	