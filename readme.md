# Proximal Neural Networks

This code belongs to the papers [1] and [2]. Please cite the corresponding paper, if you use this code.

Paper [1] is available at  
https://doi.org/10.1007/s00041-020-09761-7,  
Further the Arxiv-Preprints of the papers [1] and [2] can be found at  
https://arxiv.org/abs/1912.10480 and https://arxiv.org/abs/2011.02281.

The repository contains an implementation of Proximal Neural Networks as introduced in [1] and its convolutional counterpart as proposed in [2].
It contains scripts for reproducing the signal-denoising and classification example on the MNIST data set (http://yann.lecun.com/exdb/mnist) from [1] as well as the code for the convolutional denoiser from [2], which is trained on the BSDS data set [3] and tested on the BSD68 test set.

For questions and bug reports, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS  
2. USAGE
3. CLASSES AND FUNCTIONS
4. EXAMPLES
5. REFERENCES

## 1. REQUIREMENTS

The code requires several Python packages. We tested the code with Python 3.7.9 and the following package versions:  

Tensorflow 2.2.0  
Numpy 1.18.7  
Scipy 1.5.2  
Pillow 7.2.0  
bm3d 3.0.7  
pyunlocbox 0.5.2

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE

Download the code and import `core.stiefel_network` and `core.layers` for using the StiefelModel class.

The scripts `run_MNIST`, `run_signals` and `run_conv_denoiser` implement the numerical examples from [1] and [2]. For the usage of the classes and functions in `core.stiefel_network` and `core.layers` we refer to the corresponding examples and the documentation in Section 3.

## 3. CLASSES AND FUNCTIONS

In this section we specify the classes and functions in `core.stiefel_network` and `core.layers`

### In `core.stiefel_network`

#### class StiefelModel

This class inherits from `tensorflow.keras.Model`. It implements a proximal neural network, i.e. it consists of layers of the form A^T sigma(Ax+b) with A^T A=I or A A^T=I.

Inputs for the constructor:  
Required:

- **dHidden** - list of the numbers of neurons in the hidden layers. The lenght of dHidden specifies the number of layers. Set it to [None]\*n\_layers for the convolutional case.
- **lambdas** - list of thresholds for the soft shrinkage activation in the layers. If lambdas is a float every Layer will be generated with threshold lambdas. For other activation functions set lambda to None.

Optional:

- **activation** - activation function for the Stiefel Layers. None for soft shrinkage.
- **transposed_mat** - if false take layers of the form sigma(Ax+b) instead of A^T sigma(Ax+b). Default: True.
- **lastLayer** - True for appending an unconstrained Layer at the end (e.g. for softshrinkage). Otherwise, it should be set to None.
- **lastActivation** -  If lastLayer is None, this has no effect. Otherwise this is the Activation function for the unconstrained layer. None for softmax.
- **convolutional** - list of booleans, whether the layers are convolutional.
- **conv_shapes** - Only relevant for convolutional layers. List of the numbers of input and output filters of the convolutional layers. None for [(1,1)]\*n\_layers. 
- **filter_length** - n the convolutional case: None for full filters. If filter_length is an integer we use convolution filters of size 2\*filter_length. 
- **dim** - Dimension of the data points (1 for signals, 2 for images etc.). Default: 1.
- **fast_execution** - changes the execution order for a faster execution. Just applicable for fully convolutional networks and the weights have to be already initialized.
- **scale_layer** - Multiplicates the outputs with a scalar. Set it to False for deactivating the scaling layer, True for learning the scalar and to a float for a fixed scalar.

#### function loadData

This function loads training and test signals from a .mat file located in the data directory.  
Inputs:

- **fileName** - Path from the data directory to the file with '.mat'
- **num_data** - Number of training samples, which will be loaded.

Outputs:

- **DATA** - x_train, y_train, x_test, y_test

#### function meanPSNR

Computes the mean psnr of data of the form [:,...].
The definition of the PSNR can be found e.g. in [2].  
Inputs:

- **predictions**  - Array of shape [:,...]  
- **ground_truth** - Array of shape [:,...]  
- **one_dist**     - Set the maximal intensity to 1 (e.g. for images)

Output: PSNR

#### function MSE

Computes the mean of the L2 error along the first axis  
Inputs:

- **pred**   - predictions  
- **truth**  - ground truth  

Output: result

#### function plotSaveSignal

Easy function to plot a signal.  
Inputs:

- **signal** - Signal to plot
- **fileName** - save path
- **limits** - Limits of the axis, None for automatic scaling. Default: None.

#### function train

Implements SGD on the Stiefel manifold for minimizing the loss of a StiefelModel.  
Inputs:  
Required:

- **model** - StiefelModel which will be trained
- **x_train**, **y_train**, **x_test**, **y_test** - data

Optional:

- **EPOCHS** - Number of training epochs. Default: 5.
- **learn_rate** - learning rate. Default: 1.
- **batch_size** - batch size. Default: 32
- **loss_type** - Loss function. Can be an executable function or 'MSE' or 'crossEntropy'. Default: 'MSE'
- **show_accuracy** - True for computing the accuracy after each epoch, False for do not compute the accuracy after each epoch, None for automatic adaption based on the loss function. Default: None.
- **progress_bar** - True for showing a progress bar during training, False for hiding it. Default: True.
- **residual** - True for residual learning, False for standard learning. Default: False.

#### function train_conv

Trains a cPNN for denoising with orthogonality penalty term using the Adam optimizer.  
Inputs:  
Required:

- **model** - StiefelModel to be trained
- **epochs** - number of epochs for training
- **train_ds** - tensorflow data set containing the training data
- **backup_dir** - directory to save the weights
- **noise_level** - gaussian noise with standard deviation noise_level is added to get the corrupted data

Optional:

- **penalty** - Weight of the orthogonality penalty term. Default: 1e4

### In `core.layers`

#### function conv_mult

Multiplies two convolution filters viewed as matrices with circulant boundary conditions.  
Inputs:

- **C1** - first filter.
- **C2** - second filter.
- **full** - clarifies, if C1 and C2 have full or limited filter length. Default: False.

Output: output filter.

Calls the functions `conv_mult2D`, `conv_mult1D` or `conv_mult_full_filter`.

#### function apply_convs

Applies a convolution filter to a batch of data points.  
Inputs:

- **C** - convolution filter
- **inputs** - data points
- **boundary_contitions** - clarifies the boundary condition. Default: 'Circulant'
- **full** - clarifies, if C has full or limited filter length. Default: False.

Output: C applied on inputs.

Calls the functions `apply_convs2D`, `apply_convs1D` or `apply_convs_full_filter`.

#### function transpose_convs

Transposes convolution filter viewed as matrix.  
Inputs:

- **C** - convolution filter
- **full** - clarifies, if C has full or limited filter length. Default: False.

Output: Transposed convolution filter.

#### class StiefelDense1D

Inherits from `tensorflow.keras.layers.Layer`. This class implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function for one-dimensional data points. 
Inputs of the constructor:

- **num_outputs** - number of hidden neurons = dim(Ax).
- **soft_thresh** - threshold for the soft shrinkage function. None for learning the threshold. Only relevant if the activation function is soft shrinkage. Default: None.
- **activation** - activation function for the layer. None for soft shrinkage. Default: None
- **transposed_mat** - if false, then the Layer reads as sigma(A x+b). Default: True.

Trainable variables:

- **matrix** - matrix A.
- **bias** - bias b.
- **soft_thresh** - Only trainable, if the activation function is soft shrinkage and the threshold is set to be trainable.

#### class StiefelDense2D

Inherits from `tensorflow.keras.layers.Layer`. This class implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function for two-dimensional data points. 
Inputs of the constructor:

- **num_outputs** - number of hidden neurons = dim(Ax).
- **soft_thresh** - threshold for the soft shrinkage function. None for learning the threshold. Only relevant if the activation function is soft shrinkage. Default: None.
- **activation** - activation function for the layer. None for soft shrinkage. Default: None
- **transposed_mat** - if false, then the Layer reads as sigma(A x+b). Default: True.

Trainable variables:

- **matrix** - matrix A.
- **bias** - bias b.
- **soft_thresh** - Only trainable, if the activation function is soft shrinkage and the threshold is set to be trainable.

#### class StiefelConv1D_full

Inherits from `tensorflow.keras.layers.Layer`. This class implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a block block-circular matrix with full filters for one-dimensional data points.  
Inputs of the constructor:

- **signal_length**   - length of the input data points
- **conv_shape** - tupel with number of output and input channels
- **soft_thresh** - threshold for the soft shrinkage function. None for learning the threshold. Only relevant if the activation function is soft shrinkage. Default: None.
- **activation** - activation function for the layer. None for soft shrinkage. Default: None
- **transposed_mat** - if false, then the Layer reads as sigma(A x+b). Default: True.

Trainable variables:

- **convs** - convolution filters from the matrix A.
- **bias** - bias b.
- **soft_thresh** - Only trainable, if the activation function is soft shrinkage and the threshold is set to be trainable.

#### class StiefelConv1D

Inherits from `tensorflow.keras.layers.Layer`. This class implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a block block-circular matrix with limited filter length for one-dimensional data points.  
Inputs of the constructor:

- **filter_length**   - The size of the convolution filters is set to 2\*filter_length+1
- **conv_shape** - tupel with number of output and input channels
- **soft_thresh** - threshold for the soft shrinkage function. None for learning the threshold. Only relevant if the activation function is soft shrinkage. Default: None.
- **activation** - activation function for the layer. None for soft shrinkage. Default: None
- **transposed_mat** - if false, then the Layer reads as sigma(A x+b). Default: True.

Trainable variables:

- **convs** - convolution filters from the matrix A.
- **bias** - bias b.
- **soft_thresh** - Only trainable, if the activation function is soft shrinkage and the threshold is set to be trainable.

#### class StiefelConv2D

Inherits from `tensorflow.keras.layers.Layer`. This class implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a block block-circular matrix with limited filter length for two-dimensional data points.  
Inputs of the constructor:

- **filter_length**   - The size of the convolution filters is set to 2\*filter_length+1
- **conv_shape** - tupel with number of output and input channels
- **soft_thresh** - threshold for the soft shrinkage function. None for learning the threshold. Only relevant if the activation function is soft shrinkage. Default: None.
- **activation** - activation function for the layer. None for soft shrinkage. Default: None
- **transposed_mat** - if false, then the Layer reads as sigma(A x+b). Default: True.

Trainable variables:

- **convs** - convolution filters from the matrix A.
- **bias** - bias b.
- **soft_thresh** - Only trainable, if the activation function is soft shrinkage and the threshold is set to be trainable.

#### class ScaleLayer

Inherits from `tensorflow.keras.layers.Layer`. This layer scales the data points by a factor. If the factor in the constructor is given explicitly, then the factor is not learnable. If factor in the constructor is None, the factor will be learned.  

Input of the constructor: 

- **factor** - scaling factor.

Trainable variable:

- **factor** - Only trainable, if factor in the constructor was None.

## 4. EXAMPLES

### Signal Denoising

The script `run_signals.py` is the implementation of the denoising example in [1, Section 7]. The goal is to denoise piece-wise constant signals using
a PNN and to compare the results with the Haar basis and Haar wavelets. A detailed description of the experiment is included in [1].

### MNIST Classification

In the script `run_MNIST.py` we provide the implementation of the MNIST classification using a PNN from [1, Section 7]. A detailed
description of the experiment is included in [1].

### Convolutional Denoiser for natural images

In the example `run_conv_denoiser.py` we train a convolutional PNN for denoising natural images. As training data we use the 400 training images from the
BSDS data set. As test data we use the BSD68 data set. This reproduces the results from [2, Section 5.2]. We refer to [2] for a more
detailed description of the experiment.
Note that the training of a cPNN of the size of this example is very time consuming.

### Plug-and-play with cPNNs

In the example `run_PnP.py` we perform a FBS-PnP for denoising as well as an ADMM-PnP and FBS-PnP for deblurring. The detailed
description of the experiment is included in [2, Section 7].
Note that the training of a cPNN of the size of this example is very time consuming.

## 5. REFERENCES

[1]  M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.  
Parseval proximal neural networks.  
Journal of Fourier Analysis and Applications, 26:59, 2020.

[2] J. Hertrich, S. Neumayer and G. Steidl.  
Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.  
arXiv Preprint#2011.02281, 2020.

[3] D. Martin, C. Fowlkes, D. Tal, and J. Malik.  
A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics.  
In Proc. Eighth IEEE International Conference on Computer Vision, volume 2, pages 416–423, 2001.


