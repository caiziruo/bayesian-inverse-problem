import numpy as np
from math import e
import matplotlib.pyplot as plt

from observation import Observation

################################################################
dimension_of_observation = 100
gamma = 0.1
d = 0.4

def gaussian_kernel(t1, t2, gamma, d):
	return gamma * e ** (- ((t1 - t2) / d)**2 / 2)


################################################################

a = Observation(dimension_of_observation)
# a.plot_observation()
