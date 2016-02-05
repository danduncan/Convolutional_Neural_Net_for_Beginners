"""
This script creates and trains a convolutional neural network.
It allows the parameters of the network to be heavily customized.

Before execution, two scripts need to be run:

- First script compiles some heavily optimized cython code. Without it, runtimes would be impossibly long:
  python setup.y build_ext --inplace

- Second script downloads the CIFAR10 dataset. These are 51,000 32x32x3 images which are used for classification. 
  There are 10 total classes.
  cs231n/datasets/get_datasets.sh

For execution, you should be familiar with several implementations:
cs231n/solver.py              - Solver class
cs231n/classifiers/convnet.py - FancyNet class
cs231n/layers.py              - Forward and backward propagation algorithms for convnets
cs231n/layer_utils.py         - API's for common layer combinations
cs231n/optim.py               - Optimization algorithms used by the solver


The basic idea of the network:
Solver object will hand data to the network object, and the network returns its accuracy and the
gradients of its accuracy with respect to its internal parameters (thousands or millions).
The solver takes the parameter values and gradients and passes that information to an
optimization algorithm, which updates them by stepping each down their gradients
This repeats until time runs out or the network converges.

Runtimes (on a macbook pro):
- Small network (1 small conv layer and 1 small hidden layer) takes about ~.001 seconds / image
- Big network (2 big conv layers and 2 big hidden layers) takes about .4 seconds / image
- For a full network, an epoch is 490 iterations, assuming 100 samples per batch, and takes about 30 minutes
- 95 percent training accuracy and 72 percent validation accuracy was achieved in 10 epochs, or about 5 hours of training
- For comparison, training the full network for just 5 minutes (80 iterations) got 45 percent validation accuracy

Architecture for best results so far (72 percent validation accuracy, 5 hour training time):
Conv - SBN - Relu - Conv - SBN - Relu - Maxpool - Affine - Batchnorm - Relu - Affine - Softmax

Layer Parameters:
Conv1 - 64 7x7 filters, stride 1
Conv2 - 64 5x5 filters, stride 1
Maxpool - 2x2, stride 2
Affine1 - 2000 neurons
Affine2 - 2000 neurons
Learning rate = 1e-2
Momentum = .9 (for sgd_momentum algorithm)

Bugs:
Standalone script may have trouble running. But in the main directory, running "ipython notebook" will
open an ipython interface in your browser where the code executes.
There is a bug in sgd_momentum that prevents the user from using convolutional layers of multiple sizes.
So for now, all convolutional layers have to be the same size.

Code info:
Author: Dan Duncan
Date: 2/5/2016
Purpose:
This was part of assignment 2 in Stanford CS231n - Convolutional Neural Networks for Computer Vision

"""


"""
Import external resources
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from cs231n.classifiers.convnet import *
from cs231n.classifiers.cnn import *
from cs231n.layers import *
from cs231n.fast_layers import *

# Variables used only by ipython notebook
#%matplotlib inline # Used if you want to display in ipython notebook
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload # Keyword for ipython notebook
#%autoreload 2 # Keyword for ipython notebook

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Load preprocessed CIFAR10 data
"""NOTE: You must execute cs231n/datasets/get_datasets.sh before this willl work!"""
data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape


# Initialize these variables only if they haven't been initialized
# This allows the script to be rerun without losing information

try:
    Best_val_acc
except NameError:
    # Variables do not exist. Initialize them
    print 'Initializing Best_* variables...'
    Best_model = None;
    Best_val_acc = 0.0;
    Best_train_acc = 0.0;
else:
    print 'Best_* variables already initialized'
    pass

###############################################################
""" USER INPUTS GO BELOW """
###############################################################

# Set model hyperparameters
nfilter = [64,64];     # Number of filters (vector)
sfilter = [7,5];      # Size of filters in each layer (vector)
mp = [False,True];        # Whether maxpool is present in each layer (boolean vector)
sbn = True;        # Spatial batch norm enabled
hd = [2000,2000];         # Dimensions of hidden layers (vector)
bn = True;         # Batch norm enabled 
rg = 0.001;           # Regularization strength
ws = 1e-3;          # Weight scale

# Set solver hyperparameters
lrd = .9999;          # Learning rate decay
lr = 1e-2;          # Learning rate
mm = .9;           # Momentum (0 to 1, 0=sgd)   CHECK
uprule = 'sgd_momentum'; # Update rule

bs = 100;            # Minibatch size
ne = 10;            # Number of epochs
vb = True;          # Verbose?
pe = 200;           # Print every


# Batch together parameters for update rule
# Note that additional hyperparameters may need to be added
# if you change to a different update rule
opcon = {'learning_rate': lr, 'momentum': mm};

###############################################################
""" NO USER INPUTS PAST THIS LINE """
###############################################################


# Initialize model and solver
model = FancyNet(num_filters=nfilter, filter_sizes=sfilter, maxpools=mp,
                use_spatial_batchnorm=sbn, hidden_dims=hd, use_batchnorm=bn, reg=rg, weight_scale=ws);

solver = Solver(model, data, num_epochs=ne, batch_size=bs, update_rule=uprule, optim_config=opcon, lr_decay=lrd,
               verbose=vb, print_every=pe);

# Optimize the model (this is the part that takes a while)
solver.train();


# Check if this beats the previous best accuracy:
train_hist = solver.train_acc_history;
val_hist = solver.val_acc_history;
train_best = np.max(train_hist);
val_best = np.max(val_hist);

print 'Max training accuracy: ' + str(train_best);
print 'Max validation accuracy:  ' + str(val_best);
if val_best > Best_val_acc:
    Best_model = model;
    Best_train_acc = train_best;
    Best_val_acc = val_best;
    print '  New record!'


# Plot results
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()