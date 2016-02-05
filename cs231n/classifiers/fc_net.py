import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim);
    self.params['b1'] = np.zeros(hidden_dim);
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes);
    self.params['b2'] = np.zeros(num_classes);

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    # Get network parameters
    W1 = self.params['W1']; b1 = self.params['b1']; W2 = self.params['W2']; b2 = self.params['b2'];
    #reg = self.reg;

    # Forward pass through first layer (affine-relu)
    v1, cache1 = affine_relu_forward(X, W1, b1);

    # Forward pass through second layer (affine only)
    v2, cache2 = affine_forward(v1, W2, b2);

    # Compute scores
    scores = v2;

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # Finish forward pass of layer 2 (softmax), including gradient:
    # loss gets regularized here
    # Each separate gradient must be regularized when it is calculated
    loss, dv2 = softmax_loss(scores, y); # Softmax loss and gradient without regularization
    reg = self.reg;
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2)); # Regularize
    

    # Do backward pass through layer 2 affine
    # Get unregularized gradients 
    dv1, dw2, db2 = affine_backward(dv2, cache2);

    # Regularize the weight gradient
    dw2 += reg*W2;

    # Do backward pass through layer 1
    # Get unregularized gradients
    dx, dw1, db1 = affine_relu_backward(dv1, cache1);

    # Regularize the weight gradient
    dw1 += reg*W1;

    # Store all gradients in grads
    grads['W1'] = dw1;
    grads['b1'] = db1;
    grads['W2'] = dw2;
    grads['b2'] = db2;

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


# Utility method for layer with batch_norm
# affine --> batch norm --> relu
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


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    num_hidden = len(hidden_dims); # Number of hidden layers (does not include output layer)

    # Initialize weights and biases of each hidden layer
    
    for i in range(num_hidden):
      # Strings W1...WH and b1...bH
      wstr = 'W' + str(i+1);
      bstr = 'b' + str(i+1);

      # Size of current hidden layer
      dimw1 = hidden_dims[i];

      # Get W and b dimensions
      if (i ==0):
        dimw0 = input_dim;
      else:
        dimw0 = hidden_dims[i-1];
      
      # Initialize W and b:
      # W is size [dimw0 x dimw1]
      # b is size (dimw1,)
      self.params[wstr] = weight_scale * np.random.randn(dimw0, dimw1);
      self.params[bstr] = np.zeros(dimw1);

      # If using batch_norm, need to initialize gamma and beta
      if self.use_batchnorm:
        # Initialize gamma and beta
        # Both are same size as the current hidden layer
        self.params['gamma' + str(i+1)] = np.ones(dimw1);
        self.params['beta'  + str(i+1)] = np.zeros(dimw1);
      # No batchnorm is ever used on output layer
      

    # Initialize weight and bias for output layer
    # W is size (hN x num_classes) where hN is the size of the last hidden layer
    # b is size (num_classes,)
    self.params['W' + str(num_hidden+1)] = weight_scale * np.random.randn(hidden_dims[-1],num_classes);
    self.params['b' + str(num_hidden+1)] = np.zeros(num_classes);


    


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    num_hidden = self.num_layers - 1;
    
    # Variables to hold intermediate outputs
    v = [];     # Holds each layer's output in v[0] - v[num_layers - 1]
    cache = []; # Holds each layer's cache in cache[0] - cache[num_layers - 1]

    # Iterate over hidden layers
    for i in range(num_hidden):
      # Get hidden layer weights
      w = self.params['W' + str(i+1)]; # W1, W2, ...
      b = self.params['b' + str(i+1)]; # b1, b2 , ...

      if self.use_batchnorm:
        # Affine, then batch norm, then relu
        # Need to also get gamma and beta
        gamma = self.params['gamma' + str(i+1)]; # gamma1, gamma2, ...
        beta  = self.params['beta'  + str(i+1)]; # beta1,  beta2,  ...
        
        if i == 0:
          # Input data to first layer
          vtmp, cachetmp = affine_bn_relu_forward(X, w, b, gamma, beta, self.bn_params[i]);
        else:
          vtmp, cachetmp = affine_bn_relu_forward(v[-1], w, b, gamma, beta, self.bn_params[i]);
        
        #v.append(vtmp);
        #cache.append(cachetmp);
      else:
        # Affine forward followed by ReLU forward
        if i == 0:
          a, fc_cache = affine_forward(X, w, b);      # Input data to first layer
        else:
          a, fc_cache = affine_forward(v[-1], w, b); # Previous layer's output to deeper layers

        # ReLU forward pass
        vtmp, relu_cache = relu_forward(a); # Hidden layer outputs go in v[0] - v[num_layers - 2]
        #v.append(vtmp);

        # Store intermediate values
        #cache.append((fc_cache, relu_cache)); # Hidden layer caches go in v[0] - v[num_layers - 2]
        cachetmp = (fc_cache, relu_cache);

      # Do dropout if parameter is set
      if self.use_dropout:
        vtmp, drop_cache = dropout_forward(vtmp, self.dropout_param);
        cachetmp = (cachetmp, drop_cache);

      v.append(vtmp);
      cache.append(cachetmp);


    # Do output layer calculation
    # num_hidden = num_layers - 1
    w = self.params['W' + str(self.num_layers)]; # WN
    b = self.params['b' + str(self.num_layers)]; # bN
    # Output layer output goes in v[num_layers - 1]
    # Output layer's cache goes in cache[num_layers - 1]
    vtmp, ctmp = affine_forward(v[-1], w, b);
    v.append(vtmp);
    cache.append(ctmp);

    # Compute scores = output of the output layer
    scores = v[-1];
      
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    # Calculate loss and gradient from output layer
    # loss gets regularized here
    # Each separate gradient must be regularized when it is calculated
    loss, dout = softmax_loss(scores, y); # Loss and gradient without regularization
    reg = self.reg;

    # Iterate over all W's to add their norms to loss regularization
    for i in range(self.num_layers):
      w = self.params['W' + str(i+1)];
      loss += 0.5 * reg * np.sum(w*w);

    # Do backward pass through output layer
    # Get unregularized gradients
    # dv = dL/dvN; dw = dL/dWN; db = dL/dbN
    # dv gets fed to previous layer later
    dv, dw, db = affine_backward(dout, cache[-1]);

    # Regularize the weight gradient
    w = self.params['W' + str(self.num_layers)];
    dw += reg * w;

    # Store gradients in grads
    grads['W' + str(self.num_layers)] = dw;
    grads['b' + str(self.num_layers)] = db;

    # Iterate backward over remaining layers
    for j in range(num_hidden):
      i = num_hidden - 1 - j; # Current layer number

      cache0 = cache[i];

      if self.use_dropout:
        # In this case, cache = ((tuple of other caches), drop_cache)
        cache0, drop_cache = cache0;

        # Backprop through dropout layer
        dv = dropout_backward(dv,drop_cache);

      if self.use_batchnorm:
        # Get all gradients
        # And update dv
        #cache0 = cache[i];
        dv, dw, db, dgamma, dbeta = affine_bn_relu_backward(dv,cache0);

        # Regularize dw
        w = self.params['W' + str(i+1)];
        dw += reg * w;

        # Update grads
        grads['W' + str(i+1)] = dw;
        grads['b' + str(i+1)] = db;
        grads['gamma' + str(i+1)] = dgamma;
        grads['beta' + str(i+1)] = dbeta;

      else:

        # Backward over ReLU and affine separately
        #fc_cache, relu_cache = cache[i];
        fc_cache, relu_cache = cache0;
        da = relu_backward(dv, relu_cache);
      
        dv, dw, db = affine_backward(da, fc_cache); # Updates dv

        # Regularize dw
        w = self.params['W' + str(i+1)];
        dw += reg * w;

        # Update grads
        grads['W' + str(i+1)] = dw;
        grads['b' + str(i+1)] = db;

    # Loss and grads get returned

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
