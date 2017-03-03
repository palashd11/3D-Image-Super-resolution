import tensorflow as tf

def convolution(input, filter):
	#inputs are kernel and high-frequency image
	#output is the convolution vector or feature map
		
	#tensorflow : Computes sums of N-D convolutions (actually cross-correlation).		
	#tf.nn.convolution(input, filter, padding, strides=None, dilation_rate=None, name=None, data_format=None)

	#tensorflow : Computes a 2-D convolution given 4-D input and filter tensors.
	return tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

	#tensorflow : Depthwise 2-D convolution
	#tf.nn.depthwise_conv2d(input, filter, strides, padding, rate=None, name=None)
