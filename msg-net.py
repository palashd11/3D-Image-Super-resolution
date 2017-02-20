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
#L2 Loss.Computes half the L2 norm of a tensor without the sqrt: output = sum(t ** 2) / 2
loss = tf.nn.l2_loss(t, name=None)

#The loss is minimized using stochastic gradient descent


#set learning rate
learning_rate = 0.001


#optimizer : Stochastic Gradient descent
#Create a new Optimizer.
#Construct a new gradient descent optimizer.
#tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent') {:#GradientDescentOptimizer.init}


#Minimizing loss
tf.train.GradientDescentOptimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)



