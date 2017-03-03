import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt



#Load/Import 3d data
from somewhere import depth_input_data
DATASET_3D = depth_input_data.read_data_sets('3D_dataset') 

#Load/Import 2d data

from someplace import intensity-input_data
DATASET_2D = intensity_input_data.read_data_sets('2D_dataset')


#Placeholders for inputing data 

x=tf.placeholder(tf.float32, shape=[])

#Define variables

W=tf.Variable()
b=tf.Variable()

#Intialize variables





#two branches : intensity branch & depth/main branch

#Intensity branch
#input : high-resolution 2D image
#returns : downsampled images used for fusion in the depth branch
#relevant functions : preprocessing, feature_extraction, post_feature_extraction, convolution, pooling

#First convolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(convolution(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolution layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Third convolution layer

#Fourth convolution layer



#Depth branch
#input : low-resolution depth map(3d-image)
#output : super-resolution depth map
#relevant functions : all


#First convolution

#First deconvolution



#define loss function
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










if __name__ == '__main__': 


