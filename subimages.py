def subimages():
	#Extract patches from images and put them in the "depth" output dimension.
	tf.extract_image_patches(images, ksizes, strides, rates, padding, name=None)

	#Returns:A Tensor. Has the same type as images. 4-D Tensor with shape [batch, out_rows, out_cols, ksize_rows * ksize_cols * 		#depth] containing image patches with size ksize_rows x ksize_cols x depth vectorized in the "depth" dimension.
