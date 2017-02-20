import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

#only one branch : depth branch
#input : low resolution depth map
#output : high resolution depth map
#relevant functions : preprocessing, feature_extraction, upsampling, reconstruction

def ms_net():
	
	preprocessing()
	
	feature_extraction()

	upsampling()

	reconstruction()

	post_reconstruction()

	
