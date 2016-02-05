import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # Reshape data to rows
  # Data dimension is now (Num_images x Num_data_points)
  xr = np.reshape(x,(x.shape[0],-1));

  # Find out = xW + b
  # Matrix multiplication requires numpy.dot
  out = xr.dot(w) + b;


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
 
  # Get reshaped x:
  # Original shape: N x d1 x d2 x ... x dk
  # New shape: NxD
  xr = np.reshape(x,(x.shape[0],-1));

  # dw = x^T * dout
  dw = (np.transpose(xr)).dot(dout); # DxN * NxM = DxM

  # dx = dout * w^T
  #      NxM  * MxD = NxD
  wt = np.transpose(w); # MxD
  dx = dout.dot(wt);    # NxD
  dx = np.reshape(dx,x.shape);

  # db = ones * dout
  #      1xN  * NxM = 1xM
  N = dout.shape[0];
  o = np.ones([1,N]); # 1xN
  db = o.dot(dout);   # 1xN * NxM = 1xM
  db = np.squeeze(db); # 1xM --> (M,)



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  # Apply ReLU
  # Clamps min(x) = 0
  out = np.copy(x);
  out[out<0] = 0;

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout;
  dx[x<0] = 0;
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    # In training, whiten using the mean/variance only of the current batch
    #              Also, update the running mean.variance
      
    # Initialize cache
    cache = {};
    cache['gamma'] = gamma;       # Shape = (D,)
    cache['epsilon'] = eps;       # Scalar
    cache['x'] = x;               # Shape = NxD


    # Mean and variance are taken for all images in the batch across a particular dimension
    statdim = 0;

    # Get mean
    xm = np.mean(x,axis=statdim); # Shape = (D,)

    # Get variance
    xv = np.var(x,axis=statdim);  # Shape = (D,)

    # Update cache
    cache['xmean'] = xm;          # Shape = (D,)
    cache['xvar']  = xv;          # Shape = (D,)

    # Update running mean and variance
    running_mean = momentum*running_mean + (1-momentum)*xm; # Shape = (D,)
    running_var =  momentum*running_var  + (1-momentum)*xv; # Shape = (D,)

    # Resize xm and xv so they can be broadcast
    xm = xm[np.newaxis,:];   # Shape = 1xD
    xv = xv[np.newaxis,:];   # Shape = 1xD

    # Get normalized x
    # Includes an epsilon term to prevent divide by zero
    xhat = (x - xm) / np.sqrt(xv + eps);   # Shape = NxD
    
    # Reshape gamma and beta so they can be broadcast over xhat
    g = gamma[np.newaxis,:];  # Shape = 1xD
    b = beta[np.newaxis,:];   # Shape = 1xD

    # Execute linear transform
    out = g*xhat + b;         # Shape = NxD

    # Update cache
    cache['xhat'] = xhat;     # Shape = NxD

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
   
    # Reshape running_mean and running_var so they can be broadcast over x
    running_mean = np.squeeze(running_mean);
    running_var = np.squeeze(running_var);
    running_mean = running_mean[np.newaxis,:];
    running_var  = running_var[np.newaxis, :];

    # Find normalized x
    xhat = (x - running_mean) / np.sqrt(running_var + eps);

    # Reshape gamma and beta so they can be broadcast over xhat
    g = gamma[np.newaxis,:];
    b = beta[np.newaxis, :];

    # Execute linear transform
    out = g*xhat + b;

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  # Get all variables
  eps  = cache['epsilon'];   # Scalar
  g    = cache['gamma'];     # (D,)
  xm   = cache['xmean'];     # (D,)
  xv   = cache['xvar'];      # (D,)
  x    = cache['x'];         # NxD
  xhat = cache['xhat'];      # NxD

  
  # Reshape (D,) --> 1xD
  g  =  g[np.newaxis,:]; # 1xD
  xm = xm[np.newaxis,:]; # 1xD
  xv = xv[np.newaxis,:]; # 1xD
  
  # dout = dL/dy   # NxD

  # All sums below computed over same axis
  # Axis=0:   NxD --> 1xD
  sumdim = 0;

  # Find dL/dxhat:
  dxhat = dout*g;  # NxD

  # Find dL/d(var^2)
  dvar = np.sum(dxhat*(x-xm)*(-0.5)*(xv)**(-1.5),axis=0);

  # Find dL/d(mean)
  N = x.shape[0];
  dm = np.sum(dxhat*-1/np.sqrt(xv+eps),axis=0)+dvar*np.sum(-2*(x-xm),axis=0)/N;

  # Find dL/dx
  dx = (dxhat / np.sqrt(xv + eps));
  dx += dvar * 2 *(x - xm) / N;
  dx += dm / N;  # NxD

  # Find dL/d(gamma)
  dgamma = np.sum(dout * xhat,axis=sumdim); # 1xD
  dgamma = np.squeeze(dgamma); # (D,)

  # Find dL/d(beta)
  dbeta = np.sum(dout,axis=sumdim); # 1xD
  dbeta = np.squeeze(dbeta); # (D,)
 

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################

  # Get variables out
  eps = cache['epsilon']; # Scalar
  g   = cache['gamma'];   # (D,)
  xv  = cache['xvar'];    # (D,)
  xm  = cache['xmean'];   # (D,)
  x   = cache['x'];       # NxD
  xhat = cache['xhat'];   # NxD

  N = x.shape[0];

  # Reshape for broadcasting
  g  =  g[np.newaxis,:]; # 1xD
  xm = xm[np.newaxis,:]; # 1xD
  xv = xv[np.newaxis,:]; # 1xD

  # Get first part of dx
  dx1 = (g/np.sqrt(xv+eps)) * (dout - np.mean(dout,axis=0));

  # Get second part of dx
  K = g/(N*(xv+eps)**1.5);
  xxm = x - xm;
  dx2 = -K*(xxm - np.mean(xxm,axis=0))*np.sum(dout*xxm,axis=0);

  # Sum
  dx = dx1 + dx2;

  # dgamma and dbeta
  dgamma = np.squeeze(np.sum(dout * xhat,axis=0));
  dbeta = np.squeeze(np.sum(dout,axis=0));

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    
    mask = (np.random.rand(*x.shape) < p) / p;
    out = x*mask;

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x;
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    dx = dout/dropout_param['p'];
    dx[mask<=0] = 0;

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def get_output_size(H,W,HH,WW,S,pad):
  """
  Given the properties of the input and the convolutional layer,
  calculate the output height and width
  Return the output in a tuple.

  Inputs (all scalars):
  W: Width of input data (before padding)
  H: Height of input data
  WW: Width of filters
  HH: Height of filters
  S: Stride length

  Output:
  (Hout, Wout) - height and width of the output of the conv layer
  """
  hnum = H - HH + 2*pad;
  wnum = W - WW + 2*pad;

  hout = hnum/S + 1;
  wout = wnum/S + 1;

  # Error checking: Verify this is a valid combination of parameters
  if (hnum % S != 0):
    print 'get_output_size(): Warning: Input and filter heights not compatible.';
  if (wnum % S != 0):
    print 'get_output_size(): Warning: Input and filter widths not compatible.';

  return hout, wout;


def myim2col(x,hout,wout,HH,WW,S):
  """
  x shape = C x H x W

  In order to process an image quickly, it is converted into a 2D matrix.
  Each column in the matrix corresponds to one kernel sample of length:
    ls = hh*ww*d
  There is one column for each entry in the output array, so the number of columns is:
    hout*wout
  Thus, the dimensions of the output matrix here are:
    [hout*wout, hh*ww*d] NO 
    [HH*WW*C, hout*wout]

  Once returned, these are paired with the weight matrix, where the weights have been reshaped to:
    [F, hh*ww*d]

  Thus, the final output of the convolution layer is the product of these matrices:
    w.dot(x) --> [F, hout*wout]

  Which is then reshaped to:
    [F, hout, wout]
  """

  # Get depth
  depth = x.shape[0];

  # Get length of each sample
  ls = depth*HH*WW;

  # Create output array
  out = np.zeros([ls, hout*wout]);  # ls x hout*wout

  count = 0; # Counts number of rows which have been filled

  # Iterate over image and convert to vectors
  for i in range(hout):
    i0 = i*S; # Begin coordinate within image along dimension 1
    i1 = i0 + HH; # End coordinate

    for j in range(wout):
      j0 = j*S; # Begin coordinate within image along dimension 2
      j1 = j0 + WW; # End coordinate

                      # d  h      w
      xs = np.reshape(x[:, i0:i1, j0:j1],(ls,)); # ls x 1

      # Add to output
      out[:,count] = xs;

      count += 1;

  return out;  # (depth * HH * WW) x (hout * wout)

def conv_forward_not_totally_naive(x, w, b, conv_param):
  """
  My attempt at making a faster forward pass by converting the image to columns

  x:  (N,C,H,W)   -->  (N, C * HH * WW,H' * W;)
  x0: (C,H,W)     -->  (C * HH * WW,H' * W;)
  w0: (F,C,HH,WW) -->  (F, C * HH * WW)

  out[i,:,:,:] = reshape(w0.dot(x0))

  Doesn't seem to work though :(
  """

  # Get variables
  F, C, HH, WW = w.shape;
  N, _, H,  W  = x.shape;
  S = conv_param['stride'];
  pad = conv_param['pad'];
  hout, wout = get_output_size(H,W,HH,WW,S,pad);
  
  # Add dimension to b
  b = b[:,np.newaxis];

  # Add padding
  x = np.pad(x,((0,0),(0,0),(1,1),(1,1)),'constant',constant_values=(0,0));

  # Reshape filters to 2D matrix
  # Uses transpose to avoid mixing up axes
  # F x C x HH x WW  -->  F x C*HH*WW 
  w0 = np.reshape(w,(F,C*HH*WW));

  # Initialize output
  out = np.zeros([N, F, hout, wout]);

  # Convert to 2D matrix
  # Iterate over N samples in batch
  N = x.shape[0];
  for i in range(N):

    # Reshape current sample
    x0 = x[i,:,:,:];            # Shape = (CxHxW)
    x0 = myim2col(x0,hout,wout,HH,WW,S);  # Shape = (HH*WW*C, hout*wout)

    # Process current sample
    v = w0.dot(x0) + b;                # Shape = (F, hout*wout)

    # Reshape processed sample and add to output
    # (F, hout*wout) --> F x hout x wout
    out[i,:,:,:] = np.reshape(v,(F,hout,wout));

    return out

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  try_faster = False;
  if try_faster:
    print 'Trying less naive approach'
    out = conv_forward_not_totally_naive(x, w, b, conv_param);
    cache = (x, w, b, conv_param)
    return out, cache


  # Get variables
  F, C, HH, WW = w.shape;
  N, _, H,  W  = x.shape;
  S = conv_param['stride'];
  pad = conv_param['pad'];
  hout, wout = get_output_size(H,W,HH,WW,S,pad);
  
  # Add padding
  xp = np.pad(x,((0,0),(0,0),(1,1),(1,1)),'constant',constant_values=(0,0));

  # Initialize output volume
  out = np.zeros([N, F, hout, wout]);

  # Populate output
  for im in range(N): # Iterate over images
    xcur = xp[im,:,:,:];  # CxHxW

    for f in range(F): # Iterate over filters
      wcur = w[f,:,:,:]; # CxHHxWW

      for ii in range(hout):  # Iterate over output height coordinates
        i0 = ii*S;
        i1 = i0 + HH;

        for jj in range(wout): # Iterate over output width coordinates
          j0 = jj*S;
          j1 = j0 + HH;

          # Get current sample
          scur = xcur[:,i0:i1,j0:j1]; # CxHHxWW

          # Element-wise multiply filter by sample volume
          out0 = wcur * scur; # 

          # Get scalar sum
          out0 = np.sum(out0);

          # Add bias for final sum
          out0 += b[f];

          # Save to output
          out[im,f,ii,jj] = out0;

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # Upack cache
  x, w, b, conv_param = cache;

  # Get variables
  F, C, HH, WW = w.shape;
  N, _, H,  W  = x.shape;
  S = conv_param['stride'];
  pad = conv_param['pad'];
  hout, wout = get_output_size(H,W,HH,WW,S,pad);

  # Add padding
  x = np.pad(x,((0,0),(0,0),(1,1),(1,1)),'constant',constant_values=(0,0));

  # Initialize dx, dw, and db
  dx = np.zeros_like(x);
  dw = np.zeros_like(w);
  db = np.zeros_like(b);

  # Populate dx
  for im in range(N): # Iterate over images
    #xcur = x[im,:,:,:];  # CxHxW

    for f in range(F): # Iterate over filters
      wcur = w[f,:,:,:]; # CxHHxWW

      for ii in range(hout):  # Iterate over output height coordinates
        i0 = ii*S;
        i1 = i0 + HH;

        for jj in range(wout): # Iterate over output width coordinates
          j0 = jj*S;
          j1 = j0 + HH;

          # Populate dx
          pdx = wcur * dout[im,f,ii,jj]; # Get current filter times a particular entry in dout
          dx[im,:,i0:i1,j0:j1] += pdx; # Add this value to dx

          # Populate dw
          pdw = x[im,:,i0:i1,j0:j1] * dout[im,f,ii,jj];
          dw[f,:,:,:] += pdw;

          # Populate db
          db[f] += dout[im,f,ii,jj];

  # Strip out padding from dx
  dx = dx[:,:,pad:-pad,pad:-pad];


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  
  # Get parameters
  N, C, H, W = x.shape;
  HH = pool_param['pool_height'];
  WW = pool_param['pool_width'];
  S = pool_param['stride'];

  # Calculate output dimensions
  hnum = (H - HH); wnum = (W - WW)
  hout = hnum/S + 1;
  wout = wnum/S + 1;

  # Error checking
  if hnum%S != 0 or wnum%S != 0:
    print 'max_pool_forward_naive(): Warning: Incompatible pool and input dimensions'

  # Initialize output
  out = np.zeros([N,C,hout,wout]);

  # Populate output
  for im in range(N): # Iterate over images
    xcur = x[im,:,:,:];

    for ii in range(hout): # Iterate over output height
      i0 = ii * S;
      i1 = i0 + HH;

      for jj in range(wout): # Iterate over output width
        j0 = jj * S;
        j1 = j0 + WW;

        for kk in range(C): # Iterate over depth
          
          out[im,kk,ii,jj] = np.max(x[im,kk,i0:i1,j0:j1]);


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives (N, C, hout, wout)
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x (N, C, H, W)
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  # Get parameters
  x, pool_param = cache;
  N, C, H, W = x.shape;
  HH = pool_param['pool_height'];
  WW = pool_param['pool_width'];
  S = pool_param['stride'];

  # Calculate output dimensions
  hnum = (H - HH); wnum = (W - WW)
  hout = hnum/S + 1;
  wout = wnum/S + 1;

  # Error checking
  if hnum%S != 0 or wnum%S != 0:
    print 'max_pool_forward_naive(): Warning: Incompatible pool and input dimensions'

  # Set up dx
  dx = np.zeros_like(x);

  # Populate dx
  for im in range(N): # Iterate over images
    xcur = x[im,:,:,:];

    for ii in range(hout): # Iterate over output height
      i0 = ii * S;
      i1 = i0 + HH;

      for jj in range(wout): # Iterate over output width
        j0 = jj * S;
        j1 = j0 + WW;

        for kk in range(C): # Iterate over depth
          xmini = x[im,kk,i0:i1,j0:j1];  # Generate 2D sample from x (shape = HHxWW)

          #print xmini

          idx = np.argmax(xmini); # Scalar index of maximum value in this sample

          #print idx 

          idx = np.unravel_index(idx, xmini.shape); # Tuple index of max value

          #print idx
          #print dx[im,kk,i0:i1,j0:j1];

          # Now add 1 to this location in dx
          dx[im,kk,i0+idx[0],j0+idx[1]] += dout[im,kk,ii,jj];

          #print dx[im,kk,i0:i1,j0:j1]; 
          #print ' '



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape;
  x = np.reshape(x,(-1,C));
  xn, cache = batchnorm_forward(x,gamma,beta, bn_param);
  out = np.reshape(xn,(N,C,H,W));
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape;
  dout = np.reshape(dout,(-1,C));
  dx, dgamma, dbeta = batchnorm_backward(dout,cache);
  dx = np.reshape(dx,(N,C,H,W));
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
