import tensorflow as tf

def deconvolution(value, filter, out_shapes):
	with tf.name_scope('deconv') as scope:
    		deconv = tf.nn.conv2d_transpose(input_layer, [3, 3, 1, 1], [1, 26, 20, 1], [1, 2, 2, 1], padding='SAME', name=None)

	#The transpose of conv2d.This operation is sometimes called "deconvolution" after Deconvolutional Networks, but is actually 		the transpose (gradient) of conv2d rather than an actual deconvolution.
	return tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format='NHWC', name=None)
