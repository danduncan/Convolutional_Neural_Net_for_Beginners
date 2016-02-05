# FancyNet!

import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class FancyNet(object):
	"""

	Class containing arbitrary number of convolutional layers 
	followed by fully connected network with arbitrary number 
	of layers.

	Convolutional layers are structured like:
	
	{Conv - [batch norm] - relu - [[maxpool]]} x L1

		batch norm is optional, but if turned on, is on for the whole network
		2x2 maxpool is optional and can be specified per layer via a boolean vector 

	Fully connected layer looks like:

	{affine - [batch norm] - relu} x (L2 - 1) - affine - softmax

	
	Parameters are still stored in the self.params dictionary
	"""

	def __init__(self, input_dim=(3,32,32), num_classes=10, 
				 num_filters=[32], filter_sizes=[7], maxpools=[False], use_spatial_batchnorm=False,
				 hidden_dims=[100], use_batchnorm=False,
				 reg=0.0, weight_scale=1e-2, dtype=np.float32):

		"""
		Initialize a new FancyNet
		Implicit network hyperparameters:
			Nc - Number of convolutional layers
			Nh - Number of fully-connected hidden layers (does not include output layer)
			Padding - Every convolutional input is padded such that the output is always the same size
			Stride - all filter strides are 1

		Inputs:
		- input_dim: A tuple with the shape of the input e.g. (C,H,W)
		- num_classes: An integer giving the number of output classes to classify

		- num_filters: List of integers; number of filters in each convolutional layer; shape = (Nc,)
		- filter_sizes: List of integers; Use FSi x FSi size filter in layer i; shape = (Nc,)
		- maxpools: List of booleans; maxpools[i]==True --> use maxpool after layer i; shape = (Nc,)
		- use_spatial_batchnorm: Boolean; True --> Use spatial batchnorm on all layers

		- hidden_dims: List of integers; number of neurons in each layer of FC network; shape = (Nf,)
		- use_batchnorm: Boolean; == true --> use batchnorm on all FC layers

		- reg: Scalar giving L2 regularization strength
		- weight_scale: Scalar giving standards deviation for random initialization of weight_scale
		- dtype: float32 is faster, but float64 is more accurate (good for comparing to numerical gradients)

		Properties:
		params dictionary contains:
			Wc1, Wc2, ... 		Weights for convolutional layers
			bc1, bc2, ... 		Biases for convolutional layers
			Wf1, Wf2, ... 		Weights for fully-connected layers
			bf1, bf2, ... 		Biases for fully-connected layers
		sbn_params:  List of dictionaries of running means and variances for all spatial BN layers
		bn_params:  List of dictionaries of running means and variances for all BN layers

		"""
		self.params = {};
		self.use_spatial_batchnorm = use_spatial_batchnorm;
		self.use_batchnorm = use_batchnorm;
		self.reg = reg;
		self.Nc = len(num_filters); # Number of convolutional layers
		self.Nh = len(hidden_dims); # Number of hidden layers; Does not include output layer
		self.maxpools = maxpools;
		self.dtype = dtype;

		# Error checking
		assert (len(num_filters)==len(filter_sizes)),"Conv layer init: Vector lengths do not match!"
		assert (len(num_filters)==len(maxpools)),"Conv layer init: Vector lengths do not match!"

		# Initialize weights and biases for convolutional network
		C, H, W = input_dim;
		S = 1; # Stride for all filters


		C0, H0, W0 = C, H, W;  # Input size changes for multiple layers
		for i in range(self.Nc):

			# Padding for input, produces output of same size
			pad = (filter_sizes[i] - 1) / 2;

			# Get number of filters and filter size
			fnum = num_filters[i];
			fsize = filter_sizes[i];

			# Filters in current conv layer
			self.params['Wc' + str(i+1)] = weight_scale * np.random.randn(fnum, C0, fsize, fsize);
			self.params['bc' + str(i+1)] = np.zeros(fnum);

			# Add spatial batch normalization
			if self.use_spatial_batchnorm:
				self.params['sgamma' + str(i+1)] = np.ones(fnum); # Same shape as depth of output
				self.params['sbeta' +  str(i+1)] = np.zeros(fnum);

			# Update C0 for next layer
			C0 = num_filters[i];

			# Update height and width for next layer:
			H0 = (H0 - fsize + 2*pad)/S + 1;
			W0 = (W0 - fsize + 2*pad)/S + 1;

			# 2x2 maxpool halves both height and width
			if maxpools[i]==True:
				H0 /= 2;
				W0 /= 2;


		# Determine size of final convolutional output from final values of C0, H0, and W0:
		inLength = C0*H0*W0;

		# Initialize weights and biases for fully-connected network
		for i in range(self.Nh):
			outLength = hidden_dims[i];

			# Wf1, Wf2, ...   and  bf1, bf2, ...
			self.params['Wh' + str(i+1)] = weight_scale * np.random.randn(inLength,outLength);
			self.params['bh' + str(i+1)] = weight_scale * np.zeros(outLength);

			# If using batch_norm, need to initialize gamma and beta
			# Both are same size as current hidden layer
			if self.use_batchnorm:
				self.params['gamma' + str(i+1)] = np.ones(outLength);
				self.params['beta' + str(i+1)] = np.zeros(outLength); 

			inLength = outLength; # Update inLength for next loop

		# Initialize weight and bias for output layer
		# W is size (hN x num_classes) where hN is the size of the last hidden layer
		# b is size (num_classes,)
		self.params['Wh' + str(self.Nh+1)] = weight_scale * np.random.randn(hidden_dims[-1],num_classes);
		self.params['bh' + str(self.Nh+1)] = np.zeros(num_classes);

		# Spatial batch normalization (SBN) for conv network: Track running means and variances
		# with sbn_param to each SBN layer. Pass self.sbn_params[0] to the forward pass of the
		# first SBN layer. self.bn_params[1] to the forward pass of the SBN layer, etc.
		self.sbn_params = []
		if self.use_spatial_batchnorm:
			self.sbn_params = [{'mode': 'train'} for i in xrange(self.Nc)]

		# Batch normalization (BN) for FC network: Track running means and variances
		# with bn_param object to each BN layer. Pass self.bn_params[0] to forward pass 
		# of the first BN layer, self.bn_params[1] to forward pass of second BN layer, etc.
		self.bn_params = []
		if self.use_batchnorm:
			self.bn_params = [{'mode': 'train'} for i in xrange(self.Nh)]

		# Cast all parameters to the correct datatype
		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
  		"""
  		Compute loss and gradient for the convolutional and fully-connected nets
  		Input:  X: is shape (N, C, H, W)
  				y: is shape (C,) [optional]

	  	Returns:
  			If no y is provided, simply returns class scores and no gradients
  			If y is provided, returns loss and grads, where grads is a dictionary of gradients
  		"""

		X = X.astype(self.dtype);
		N, C, H, W = X.shape;
		mode = 'test' if y is None else 'train'

		# Set the train/test norm for all SBN and BN params
		if self.use_spatial_batchnorm:
			for sbn_param in self.sbn_params:
				sbn_param['mode'] = mode;
  	
		if self.use_batchnorm:
			for bn_param in self.bn_params:
				bn_param['mode'] = mode;
  	
		scores = None;

		# Variables to hold intermediate outputs
		vcon = []; 	   # All convolutional outputs
		cachecon = []; # All convolutional caches
		vfc = []; 	   # All FC outputs
		cachefc = [];  # All FC caches

		# Dictionary to be passed to any maxpool layers:
		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		# Iterate over convolutional layers
		vprev = X; # Input to first layer is X, will be v[-1] in later layers
		for i in range(self.Nc):
			wc = self.params['Wc' + str(i+1)];   # F x C x HH x WW
			bc = self.params['bc' + str(i+1)];   # (F,)

			# Dictionary to pass to conv layer:
			filter_size = wc.shape[2];
			conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

			# Check whether to use the maxpool and/or spatial batch norm
			if self.use_spatial_batchnorm:
				sgamma = self.params['sgamma' + str(i+1)];
				sbeta = self.params['sbeta' + str(i+1)];

				if self.maxpools[i]: # Batchnorm and maxpool
					vtmp, cachetmp = conv_sbn_relu_pool_forward(vprev, wc, bc, sgamma, sbeta, 
	  								  		conv_param, self.sbn_params[i], pool_param);
				else: # Batchnorm yes, but no maxpool
					vtmp, cachetmp = conv_sbn_relu_forward(vprev, wc, bc, sgamma, sbeta, 
		  									conv_param, self.sbn_params[i]);
			elif self.maxpools[i]: # Yes maxpool, but no spatial batchnorm
				vtmp, cachetmp = conv_relu_pool_forward(vprev, wc, bc, conv_param, pool_param);
			else: # No SBN or maxpool
				vtmp, cachetmp = conv_relu_forward(vprev, wc, bc, conv_param);


		  	# Update vcon, cachecon, and vprev
			vcon.append(vtmp);
			cachecon.append(cachetmp);
			vprev = vcon[-1];

		### FULLY CONNECTED LAYERS ###
		v_conv_shape = vcon[-1].shape; # Save this value for use in reshaping during backprop
		infc = vcon[-1]; # First input is the output of the last convolutional layer
		infc = np.reshape(infc,(N,-1)); # Reshape to 2D: shape is now: N x (C*HH*WW) 
		  								  # N is number of samples

		# Iterate over fully-connected hidden layers
		for i in range(self.Nh):
			# Get hidden layer weights and biases
			wh = self.params['Wh' + str(i+1)];
			bh = self.params['bh' + str(i+1)];

			if self.use_batchnorm:
				gamma = self.params['gamma' + str(i+1)];
				beta = self.params['beta' + str(i+1)];

				#print 'infc size = ' + str(infc.shape) + '   wh shape = ' + str(wh.shape) + '  gamma shape = ' + str(gamma.shape);

				vtmp, cachetmp = affine_bn_relu_forward(infc, wh, bh, gamma, beta, self.bn_params[i]);
				#print 'vtmp shape = ' + str(vtmp.shape);
				#vtmp = np.squeeze(vtmp);
			else:
				vtmp, cachetmp = affine_relu_forward(infc, wh, bh);

			# Save output and cache
			vfc.append(vtmp);
			cachefc.append(cachetmp);

			# Update input infc:
			infc = vtmp;

		### OUTPUT LAYER ###
		wh = self.params['Wh' + str(self.Nh+1)];
		bh = self.params['bh' + str(self.Nh+1)];
		#print infc.shape;
		#print wh.shape;
		#print bh.shape;
		vtmp, cachetmp = affine_forward(infc, wh, bh);
		  
		vfc.append(vtmp);
		cachefc.append(cachetmp);

		# Compute scores
		scores = vfc[-1];

		# If in test mode, return early
		if mode == 'test':
			return scores;

		###############################################################################################
		###############################################################################################
		  
		### LOSS AND GRADIENTS ###
		loss, grads = 0.0, {};

		loss, dout = softmax_loss(scores, y);
		reg = self.reg;

		# Iterate over all W's to add their norms to loss regularization
		for i in range(self.Nc):
			w = self.params['Wc' + str(i+1)];
			loss += 0.5 * reg * np.sum(w*w);

		for i in range(self.Nh + 1):   # +1 makes sure output layer weight is included
			w = self.params['Wh' + str(i+1)];
			loss += 0.5 * reg * np.sum(w*w);

	  	
		# Do backward pass through output layer
		# Get unregularized gradients
		# dv = dL/dvN; dw = dL/dWN; db = dL/dbN
	   	# dv gets fed to previous layer later
	   	# pop() removes the last element from the list
		dv, dw, db = affine_backward(dout, cachefc.pop());

		# Regularize gradient and save to grads
		wh = self.params['Wh' + str(self.Nh + 1)];
		dw += reg * wh;
		grads['Wh' + str(self.Nh+1)] = dw;
		grads['bh' + str(self.Nh+1)] = db;

		# Iterate backwards over remaining fully-connected layers
		for j in range(self.Nh): 
			i = self.Nh - 1 - j; # Iterates backwards

			cache0 = cachefc.pop(); # Removes last cache from cachefc

			if self.use_batchnorm:
				dv, dw, db, dgamma, dbeta = affine_bn_relu_backward(dv,cache0);
				grads['gamma' + str(i+1)] = dgamma;
				grads['beta' + str(i+1)] = dbeta;
			else: # No batch norm
				dv, dw, db = affine_relu_backward(dv, cache0);

			# Regularize dw
			wh = self.params['Wh' + str(i+1)];
			dw += reg * wh;

			# Update grads
			grads['Wh' + str(i+1)] = dw;
			grads['bh' + str(i+1)] = db;


		### CONVOLUTIONAL LAYERS ###
		# Reshape input to match layer output
		dv = np.reshape(dv, v_conv_shape);    

		# Iterate backwards over convolutional layers
		for j in range(self.Nc):
			i = self.Nc - 1 - j; # Iterate backwards

			cache0 = cachecon.pop(); # Removes last element from cachecon

			if self.use_spatial_batchnorm:
				if self.maxpools[i]:
					# Backprop through:
					# conv <-- spatial BN <-- Relu <-- Maxpool
					dv, dw, db, dgamma, dbeta = conv_sbn_relu_pool_backward(dv, cache0);
				else:
					# Backprop through
					# conv <-- spatial BN <-- Relu
					dv, dw, db, dgamma, dbeta = conv_sbn_relu_backward(dv, cache0);

				# Save spatial BN gradients:
				grads['sgamma' + str(i+1)] = dgamma;
				grads['sbeta'  + str(i+1)] = sbeta;

			elif self.maxpools[i]:
				# Backprop through:
				# conv <-- Relu <-- maxpool
				dv, dw, db = conv_relu_pool_backward(dv, cache0);
			else:
				# Backprop through:
				# conv <-- Relu
				dv, dw, db = conv_relu_backward(dv, cache0);

			# Regularize dw
			wc = self.params['Wc' + str(i+1)];
			dw += reg*wc;

			# Save gradients
			grads['Wc' + str(i+1)] = dw;
			grads['bc' + str(i+1)] = db;

		return loss, grads;


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
	a, fc_cache = affine_forward(x, w, b);
	b, bn_cache = batchnorm_forward(a,gamma,beta,bn_param);
	out, relu_cache = relu_forward(b);
	cache = (fc_cache, bn_cache, relu_cache);
	return out, cache;

def affine_bn_relu_backward(dout, cache):
	fc_cache, bn_cache, relu_cache = cache;
	da = relu_backward(dout, relu_cache);
	dbn, dgamma, dbeta = batchnorm_backward_alt(da,bn_cache);
	dx, dw, db = affine_backward(dbn, fc_cache);
	return dx, dw, db, dgamma, dbeta;
































