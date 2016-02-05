import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################    
    C, H, W = input_dim;

    # Dimensions of data output by convolutional layer
    S = 1; pad = (filter_size - 1)  / 2; # Stride and image padding
    hconv = (H - filter_size + 2*pad)/S + 1;
    wconv = (W - filter_size + 2*pad)/S + 1;

    # Get dimensions of 2x2 max-pool output
    hmp = hconv / 2;
    wmp = wconv / 2;

    # Get dimensions of vector fed into affine layer
    # Convert maxpool output by using np.reshape(v1,(N,-1))
    # Recover by using np.reshape(dv1,v1.shape)
    laff = hmp*wmp*num_filters;

    # Determine starting weight and bias matrices
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size);
    self.params['b1'] = np.zeros(num_filters);
    self.params['W2'] = weight_scale * np.random.randn(laff, hidden_dim);
    self.params['b2'] = np.zeros(hidden_dim);
    self.params['W3'] = weight_scale * np.random.rand(hidden_dim,num_classes);
    self.params['b3'] = np.zeros(num_classes);
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    N, C, H, W = X.shape;

    #print 'X shape = ' + str(X.shape);

    # Get conv layer output. Note that it is not 2-dimensional    
    # conv - relu - 2x2 maxpool
    v1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param);

    #print 'v1 shape = ' + str(v1.shape);

    # Reshape to 2D
    v1shape = v1.shape; # Used to reshape back to original form in backward pass
    v1 = np.reshape(v1,(N,-1));
    #print 'v1 shape = ' + str(v1.shape);

    # Feed forward to hidden layer (affine-relu)
    v2, cache2 = affine_relu_forward(v1, W2, b2);
    #print 'v2 shape = ' + str(v2.shape);

    # Feed forward to final layer (affine only)
    v3, cache3 = affine_forward(v2, W3, b3)
    #print 'v3 shape = ' + str(v3.shape);

    # Compute scores
    scores = v3;

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    # Calculate softmax loss from layer 2 output
    # Loss gets regularized here
    # Each separate gradient must be regularized later when calculated
    loss, dv3 = softmax_loss(scores,y); # Softmax loss and gradient
    #print 'dv3 shape = ' + str(dv3.shape);
    reg = self.reg;
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)); # Regularize

    # Do backward pass through layer 2 affine
    dv2, dw3, db3 = affine_backward(dv3, cache3);
    dw3 += reg*W3; # Regularize
    #print 'dv2 shape = ' + str(dv2.shape);


    # Backward pass through hidden layer
    dv1, dw2, db2 = affine_relu_backward(dv2, cache2);
    dw2 += reg*W2;  # Regularize
    #print 'dv1 shape = ' + str(dv1.shape);

    # Reshape dv1 to be compatible with convolutional layer
    dv1 = np.reshape(dv1,v1shape);
    #print 'dv1 shape = ' + str(dv1.shape);

    # Do backward pass through convolutional layer
    dx, dw1, db1 = conv_relu_pool_backward(dv1, cache1);
    dw1 += reg*W1; # Regularize

    # Store all weight and bias gradients in grads
    grads['W1'] = dw1; grads['b1'] = db1;
    grads['W2'] = dw2; grads['b2'] = db2;
    grads['W3'] = dw3; grads['b3'] = db3;





    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  


  
pass
