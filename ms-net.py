import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

def ms_net():
	
	preprocessing()
	
	feature_extraction()

	upsampling()

	reconstruction()

	post_reconstruction()

	
