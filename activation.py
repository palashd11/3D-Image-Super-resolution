def activation():
	#tensorflow : Computes rectified linear: max(features, 0).
	tf.nn.relu(features, name=None)
	
	#returns : A Tensor. Has the same type as features
