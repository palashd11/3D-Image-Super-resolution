import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

#two branches : intensity branch & depth/main branch

#Intensity branch
#input : high-resolution 2D image
#returns : downsampled images used for fusion in the depth branch
#relevant functions : preprocessing, feature_extraction, post_feature_extraction, convolution, pooling




#Depth branch
#input : low-resolution depth map(3d-image)
#output : super-resolution depth map
#relevant functions : all




#define cost function
#set learning rate
#set optimizer
#minimize cost function
