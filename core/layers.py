# This code belongs to the papers
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# and
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.
#
# Please cite the corresponding paper if you use this code.
#
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
import numpy as np
import numpy.matlib

def conv_mult(C1,C2,full=False):
    # Multiplies two convolution filters with circulant boundary conditions.
    # Inputs:
    #     - C1       - first filter.
    #     - C2       - second filter.
    #     - full     - clarifies, if C1 and C2 have full or limited filter length.
    #    
    # Output: output filter
    #
    if full:
        return conv_mult_full_filter(C1,C2)
    elif len(C1.shape)==4:
        return conv_mult2D(C1,C2)
    elif len(C1.shape)==3:
        return conv_mult1D(C1,C2)
    else:
        raise ValueError('C1 has to be a 2D or 3D-Convolution kernel!')

def conv_mult2D(C1,C2):
    filter_length=(C1.shape[2]-1)//2
    #C_r=tf.reverse(C2,[2])
    C_r=tf.reverse(C1,[2])
    C_r=tf.reverse(C_r,[3])
    #C_inp=tf.transpose(C_r,[1,2,3,0])
    C_inp=tf.transpose(C2,[1,2,3,0])
    C_inp_long=tf.concat([tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2],C_inp.shape[3])),C_inp,tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2],C_inp.shape[3]))],1)
    C_inp_long=tf.concat([tf.zeros((C_inp_long.shape[0],C_inp_long.shape[1],filter_length,C_inp_long.shape[3])),C_inp_long,tf.zeros((C_inp_long.shape[0],C_inp_long.shape[1],filter_length,C_inp_long.shape[3]))],2)
    #C_trans_kernel=tf.transpose(C1,[2,3,1,0])
    C_trans_kernel=tf.transpose(C_r,[2,3,1,0])
    C_1_C2_out=tf.nn.conv2d(C_inp_long,C_trans_kernel,1,'SAME')
    C_T_C=tf.transpose(C_1_C2_out,[3,0,1,2])
    return C_T_C

def conv_mult1D(C1,C2):
    filter_length=(C1.shape[2]-1)//2
    C_r=tf.reverse(C1,[2])
    C_inp=tf.transpose(C2,[1,2,0])
    C_inp_long=tf.concat([tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2])),C_inp,tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2]))],1)
    C_trans_kernel=tf.transpose(C_r,[2,1,0])
    C_1_C2_out=tf.nn.conv1d(C_inp_long,C_trans_kernel,1,'SAME')
    C_T_C=tf.transpose(C_1_C2_out,[2,0,1])
    return C_T_C

def apply_convs(C,inputs,boundary_conditions='Circulant',full=False):
    # Applies a convolution filter to a batch of data points.
    # Inputs:
    #     - C                   - convolution filter
    #     - inputs              - data points
    #     - boundary_contitions - clarifies the boundary condition. Default: 'Circulant'
    #     - full                - clarifies, if C1 and C2 have full or limited filter length. Default: False.
    #
    # Output: C applied on inputs.
    #
    if full:
        return apply_convs_full_filter(C,inputs)
    elif len(C.shape)==4:
        return apply_convs2D(C,inputs,boundary_conditions=boundary_conditions)
    elif len(C.shape)==3:
        return apply_convs1D(C,inputs,boundary_conditions=boundary_conditions)
    else:
        raise ValueError('C has to be a 2D or 3D-Convolution kernel!')

def apply_convs2D(C,inputs,boundary_conditions='Circulant'):
    filter_length=(C.shape[2]-1)//2
    if len(inputs.shape)<4:
        inputs=tf.expand_dims(inputs,3)
    if boundary_conditions=='Zero':
        kernel=tf.transpose(C,[2,3,1,0])
        x=tf.nn.conv2d(inputs,kernel,1,'SAME')
        return x
    if boundary_conditions=='Circulant':
        inputs_long=tf.concat([inputs[:,(-filter_length):,:],inputs,inputs[:,0:filter_length,:,:]],1)
        inputs_long=tf.concat([inputs_long[:,:,(-filter_length):,:],inputs_long,inputs_long[:,:,0:filter_length,:]],2)
    elif boundary_conditions=='Neumann':
        inputs_long=tf.concat([tf.reverse(inputs[:,:filter_length,:],[1]),inputs,tf.reverse(inputs[:,(-filter_length):,:,:],[1])],1)
        inputs_long=tf.concat([tf.reverse(inputs_long[:,:,:filter_length,:],[2]),inputs_long,tf.reverse(inputs_long[:,:,(-filter_length):,:],[2])],2)        
    else:
        raise ValueError('Unknown boundary conditions!')
    kernel=tf.transpose(C,[2,3,1,0])
    x=tf.nn.conv2d(inputs_long,kernel,1,'VALID')
    return x

def apply_convs1D(C,inputs,boundary_conditions='Circulant'):
    filter_length=(C.shape[2]-1)//2
    if len(inputs.shape)<3:
        inputs=tf.expand_dims(inputs,2)
    if boundary_conditions=='Zero':
        kernel=tf.transpose(C,[2,1,0])
        x=tf.nn.conv1d(inputs,kernel,1,'SAME')
        return x
    if boundary_conditions=='Circulant':
        inputs_long=tf.concat([inputs[:,(-filter_length):,:],inputs,inputs[:,0:filter_length,:]],1)
    elif boundary_conditions=='Neumann':
        inputs_long=tf.concat([tf.reverse(inputs[:,:filter_length,:],[1]),inputs,tf.reverse(inputs[:,(-filter_length):,:],[1])],1)        
    else:
        raise ValueError('Unknown boundary conditions!')
    kernel=tf.transpose(C,[2,1,0])
    x=tf.nn.conv1d(inputs_long,kernel,1,'VALID')
    return x

def conv_mult_full_filter(C1,C2):
    if len(C1.shape)==4:
        raise ValueError('C1 has to be a 2D-Convolution kernel!')
    elif len(C1.shape)==3:
        return conv_mult1D_full_filter(C1,C2)
    else:
        raise ValueError('C1 has to be a 2D-Convolution kernel!')

def apply_convs_full_filter(C,inputs):
    if len(C.shape)==4:
        raise ValueError('C has to be a 2D-Convolution kernel!')
    elif len(C.shape)==3:
        return apply_convs1D_full_filter(C,inputs)
    else:
        raise ValueError('C has to be a 2D-Convolution kernel!')

def apply_convs1D_full_filter(C,inputs):
    if len(inputs.shape)<3:
        inputs=tf.expand_dims(inputs,2)
    inputs_long=tf.concat([inputs,inputs[:,0:-1,:]],1)
    kernel=tf.transpose(C,[2,1,0])
    x=tf.nn.conv1d(inputs_long,kernel,1,'VALID')
    return x

def conv_mult1D_full_filter(C1,C2):
    C_r=tf.reverse(C1,[2])
    C_r=tf.roll(C_r,1,2)
    C_inp=tf.transpose(C2,[1,2,0])
    C_inp_long=tf.concat([C_inp,C_inp[:,:-1,:]],1)
    C_trans_kernel=tf.transpose(C_r,[2,1,0])
    C_1_C2_out=tf.nn.conv1d(C_inp_long,C_trans_kernel,1,'VALID')
    C_T_C=tf.transpose(C_1_C2_out,[2,0,1])
    return C_T_C


def transpose_convs_full_filter(C):
    C=tf.reverse(C,[2])
    if len(C.shape)==4:
        raise ValueError('C has to be a 2D-Convolution kernel!')
    C=tf.roll(C,1,2)
    C=tf.transpose(C,[1,0,2])
    return C

def transpose_convs(C,full=False):
    # Transposes convolution filter viewed as matrix.
    # Inputs:
    #     - C      - convolution filter
    #     - full   - clarifies, if C has full or limited filter length. Default: False.
    # 
    # Output: Transposed convolution filter.
    if full:
        return transpose_convs_full_filter(C)
    C=tf.reverse(C,[2])
    if len(C.shape)==4:    
        C=tf.reverse(C,[3])
        C=tf.transpose(C,[1,0,2,3])
    else:
        C=tf.transpose(C,[1,0,2])
    return C
   

class StiefelDense1D(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function for one-dimensional
    # data points.
    # Inputs:   num_outputs     = number of hidden neurons = dim(Ax)
    #           soft_thresh     = threshold for the soft shrinkage function
    #           activation      = activation function for the layer. None for soft shrinkage
    #           transposed_mat  = if false, then the Layer reads as sigma(A x+b)
    def __init__(self,num_outputs,soft_thresh=None,activation=None,transposed_mat=True):
        super(StiefelDense1D, self).__init__()
        if activation is None:
            self.soft_shrinkage_activation=True
        else:
            self.soft_shrinkage_activation=False
            self.activation=activation
        self.num_outputs=num_outputs
        if soft_thresh is None and self.soft_shrinkage_activation:
            self.learn_soft_thresh=True
        else:
            self.learn_soft_thresh=False
            self.soft_thresh=soft_thresh
        self.transposed_mat=transposed_mat
        self.conv=False


    def build(self, input_shape):
        self.matrix=self.add_weight("matrix",initializer='orthogonal',shape=[self.num_outputs,int(np.prod(input_shape[1:]))],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[self.num_outputs], trainable=True)
        if self.learn_soft_thresh:
            init_thresh=tf.constant_initializer(0.01)
            self.soft_thresh=self.add_weight("soft_thresh",initializer=init_thresh,shape=[1],trainable=True)

    def execute(self,inputs,matrix,bias,soft_thresh):
        if len(inputs.shape)==3:
            inputs=tf.concat([inputs[:,:,i] for i in range(inputs.shape[2])],axis=1)
        x=tf.linalg.matmul(matrix,tf.transpose(inputs))
        #if inputs.shape[-1] is None or inputs.shape[0] is None:
        #    return inputs
        x = nn.bias_add(tf.transpose(x), bias)
        if self.soft_shrinkage_activation:
            if self.learn_soft_thresh:
                soft_thresh=tf.exp(soft_thresh)
            x=tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-soft_thresh,0))
        else:
            x=self.activation(x)
        if self.transposed_mat:
            outputs = tf.transpose(tf.linalg.matmul(tf.transpose(matrix),tf.transpose(x)))
        else:
            outputs=x
        return outputs

    def call(self, inputs):
        return self.execute(inputs,self.matrix,self.bias,self.soft_thresh)

class StiefelDense2D(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function for two-dimensional
    # data points.
    # Inputs:   num_outputs     = number of hidden neurons = dim(Ax)
    #           soft_thresh     = threshold for the soft shrinkage function
    #           activation      = activation function for the layer. None for soft shrinkage
    #           transposed_mat  = if false, then the Layer reads as sigma(A x+b)
    def __init__(self,num_outputs,soft_thresh=None,activation=None,transposed_mat=True):
        super(StiefelDense2D, self).__init__()
        if activation is None:
            self.soft_shrinkage_activation=True
        else:
            self.soft_shrinkage_activation=False
            self.activation=activation
        self.num_outputs=num_outputs
        if soft_thresh is None and self.soft_shrinkage_activation:
            self.learn_soft_thresh=True
        else:
            self.learn_soft_thresh=False
            self.soft_thresh=soft_thresh
        self.transposed_mat=transposed_mat
        self.conv=False


    def build(self, input_shape):
        self.matrix=self.add_weight("matrix",initializer='orthogonal',shape=[self.num_outputs,int(np.prod(input_shape[1:]))],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[self.num_outputs], trainable=True)
        if self.learn_soft_thresh:
            init_thresh=tf.constant_initializer(0.01)
            self.soft_thresh=self.add_weight("soft_thresh",initializer=init_thresh,shape=[1],trainable=True)

    def execute(self,inputs,matrix,bias,soft_thresh):
        if len(inputs.shape)<4:
            inputs=tf.expand_dims(inputs,3)
        inp_shape=inputs.shape
        inputs=tf.reshape(inputs,[-1,tf.reduce_prod(inputs.shape[1:])])
        x=tf.linalg.matmul(matrix,tf.transpose(inputs))
        x = nn.bias_add(tf.transpose(x), bias)
        if self.soft_shrinkage_activation:
            if self.learn_soft_thresh:
                soft_thresh=soft_thresh
            x=tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-soft_thresh,0))
        else:
            x=self.activation(x)
        if self.transposed_mat:
            outputs = tf.transpose(tf.linalg.matmul(tf.transpose(matrix),tf.transpose(x)))
            outputs=tf.reshape(outputs,[-1,inp_shape[1],inp_shape[2],inp_shape[3]])
            if outputs.shape[3]==1:
                outputs=tf.reduce_sum(outputs,3)
        else:
            outputs=x
        return outputs

    def call(self, inputs):
        return self.execute(inputs,self.matrix,self.bias,self.soft_thresh)

class StiefelConv1D_full(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a
    # block circular matrix with full filters for one-dimensional data points.
    # Inputs:   signal_length   = length of the signal
    #           soft_thresh     = threshold for the soft shrinkage function
    #           activation      = activation function for the layer. None for softShrinkage
    #           transposed_mat  = if false, then the Layer reads as sigma(A x+b)
    #           conv_shape      = tupel with number of output and input channels
    def __init__(self,signal_length,conv_shape=(1,1),soft_thresh=None,activation=None,transposed_mat=False):
        super(StiefelConv1D_full, self).__init__()
        if activation is None:
            self.soft_shrinkage_activation=True
        else:
            self.soft_shrinkage_activation=False
            self.activation=activation
        self.signal_length=signal_length
        if soft_thresh is None and self.soft_shrinkage_activation:
            self.learn_soft_thresh=True
        else:
            self.learn_soft_thresh=False
            self.soft_thresh=soft_thresh
        self.transposed_mat=transposed_mat
        self.conv=True
        self.dim=1
        self.multi_channel=True
        self.limited_filter=False
        self.conv_shape=conv_shape


    def build(self, input_shape):
        if len(input_shape)==2:
            input_shape=(input_shape[0],input_shape[1],1)
        self.convs=self.add_weight("convs",initializer='random_normal',shape=[self.conv_shape[0],self.conv_shape[1],self.signal_length],trainable=True)
        #self.bias=self.add_weight("bias",initializer='zeros', shape=[self.signal_length,self.conv_shape[0]], trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[1,self.conv_shape[0]], trainable=True)
        if self.learn_soft_thresh:
            init_thresh=tf.constant_initializer(0.01)
            self.soft_thresh=self.add_weight("soft_thresh",initializer=init_thresh,shape=[1],trainable=True)

    def execute(self,inputs,convs,bias,soft_thresh):
        if len(inputs.shape)<3:
            inputs=tf.expand_dims(inputs,2)
        if self.conv_shape[1]>1 and inputs.shape[2]==1:
            inputs=tf.tile(inputs,(1,1,self.conv_shape[1]))
            inputs=inputs/tf.math.sqrt(tf.constant(self.conv_shape[1],dtype=tf.float32))
        x=apply_convs_full_filter(convs,inputs)
        x = tf.add(x, bias)
        if self.soft_shrinkage_activation:
            x=tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-soft_thresh,0))
        else:
            x=self.activation(x)
         
        if self.transposed_mat:
            convs_transpose=transpose_convs_full_filter(convs)
            outputs=apply_convs_full_filter(convs_transpose,x)
        else:
            outputs=x
        if outputs.shape[2]==1:
            #if outputs.shape[0] is None:
            #    return outputs
            outputs=tf.reduce_sum(outputs,2)
            #outputs=tf.reshape(outputs,(outputs.shape[0],outputs.shape[1]))
        return outputs
    
    def call(self, inputs):
        return self.execute(inputs,self.convs,self.bias,self.soft_thresh)

class StiefelConv1D(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a
    # block circular matrix with limited filter length for one dimensional data points.
    # Inputs:   filter_length   = The size of the convolution filters is set to 2\*filter_length+1
    #           soft_thresh     = threshold for the soft shrinkage function
    #           activation      = activation function for the layer. None for softShrinkage
    #           transposed_mat  = if false, then the Layer reads as sigma(A x+b)
    #           conv_shape      = tupel with number of output and input channels
    def __init__(self,filter_length=None,conv_shape=(1,1),soft_thresh=None,activation=None,transposed_mat=False):
        super(StiefelConv1D, self).__init__()
        if activation is None:
            self.soft_shrinkage_activation=True
        else:
            self.soft_shrinkage_activation=False
            self.activation=activation
        if filter_length is None:
            filter_length=2
        self.filter_length=filter_length
        if soft_thresh is None and self.soft_shrinkage_activation:
            self.learn_soft_thresh=True
        else:
            self.learn_soft_thresh=False
            self.soft_thresh=soft_thresh
        self.transposed_mat=transposed_mat
        self.conv=True
        self.dim=1
        self.multi_channel=True
        self.limited_filter=True
        self.conv_shape=conv_shape


    def build(self, input_shape):
        if len(input_shape)==2:
            input_shape=(input_shape[0],input_shape[1],1)
        init_mat=np.zeros((self.conv_shape[0],self.conv_shape[1],2*self.filter_length+1))
        #for i in range(0,min(self.conv_shape[0],self.conv_shape[1])):
        #    init_mat[i,i,self.filter_length]=1
        #init=tf.constant_initializer(init_mat)
        self.convs=self.add_weight("convs",initializer='random_normal',shape=[self.conv_shape[0],self.conv_shape[1],2*self.filter_length+1],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[1,self.conv_shape[0]], trainable=True)
        if self.learn_soft_thresh:
            init_thresh=tf.constant_initializer(0.01)
            self.soft_thresh=self.add_weight("soft_thresh",initializer=init_thresh,shape=[1],trainable=True)

    def execute(self,inputs,convs,bias,soft_thresh):
        if len(inputs.shape)<3:
            inputs=tf.expand_dims(inputs,2)
        if self.conv_shape[1]>1 and inputs.shape[2]==1:
            inputs=tf.tile(inputs,(1,1,self.conv_shape[1]))
            inputs=inputs/tf.math.sqrt(tf.constant(self.conv_shape[1],dtype=tf.float32))
        inputs_long=tf.concat([inputs[:,(-self.filter_length):,:],inputs,inputs[:,0:self.filter_length,:]],1)
        kernel=tf.transpose(convs,[2,1,0])
        x=tf.nn.conv1d(inputs_long,kernel,1,'VALID')
        x = tf.add(x, bias)
        if self.soft_shrinkage_activation:
            x=tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-soft_thresh,0))
        else:
            x=self.activation(x)
         
        if self.transposed_mat:
            x_long=tf.concat([x[:,(-self.filter_length):,:],x,x[:,0:self.filter_length,:]],1)
            kernels=tf.reverse(convs,[2])
            kernels=tf.transpose(kernels,[1,0,2])
            kernel=tf.transpose(kernels,[2,1,0])
            outputs=tf.nn.conv1d(x_long,kernel,1,'VALID')
        else:
            outputs=x
        if outputs.shape[2]==1:
            #if outputs.shape[0] is None:
            #    return outputs
            outputs=tf.reduce_sum(outputs,2)
            #outputs=tf.reshape(outputs,(outputs.shape[0],outputs.shape[1]))
        return outputs
    
    def call(self, inputs):
        return self.execute(inputs,self.convs,self.bias,self.soft_thresh)

class StiefelConv2D(tf.keras.layers.Layer):
    # Implements a Layer of the form A^T sigma(A x+ b) where sigma is an activation function and A is a
    # block circular matrix with limited filter length for two dimensional data points.
    # Inputs:   filter_length   = The size of the convolution filters is set to 2\*filter_length+1
    #           soft_thresh     = threshold for the soft shrinkage function
    #           activation      = activation function for the layer. None for softShrinkage
    #           transposed_mat  = if false, then the Layer reads as sigma(A x+b)
    #           conv_shape      = tupel with number of output and input channels
    def __init__(self,filter_length=None,conv_shape=(1,1),soft_thresh=None,activation=None,transposed_mat=False):
        super(StiefelConv2D, self).__init__()
        if activation is None:
            self.soft_shrinkage_activation=True
        else:
            self.soft_shrinkage_activation=False
            self.activation=activation
        if filter_length is None:
            filter_length=2
        self.filter_length=filter_length
        if soft_thresh is None and self.soft_shrinkage_activation:
            self.learn_soft_thresh=True
        else:
            self.learn_soft_thresh=False
            self.soft_thresh=soft_thresh
        self.transposed_mat=transposed_mat
        self.conv=True
        self.dim=2
        self.multi_channel=True
        self.limited_filter=True
        self.conv_shape=conv_shape


    def build(self, input_shape):
        if len(input_shape)==2:
            input_shape=(input_shape[0],input_shape[1],1)
        init_mat=np.zeros((self.conv_shape[0],self.conv_shape[1],2*self.filter_length+1))
        #for i in range(0,min(self.conv_shape[0],self.conv_shape[1])):
        #    init_mat[i,i,self.filter_length]=1
        #init=tf.constant_initializer(init_mat)
        self.convs=self.add_weight("convs",initializer='random_normal',shape=[self.conv_shape[0],self.conv_shape[1],2*self.filter_length+1,2*self.filter_length+1],trainable=True)
        self.bias=self.add_weight("bias",initializer='zeros', shape=[1,self.conv_shape[0]], trainable=True)
        if self.learn_soft_thresh:
            init_thresh=tf.constant_initializer(0.01)
            self.soft_thresh=self.add_weight("soft_thresh",initializer=init_thresh,shape=[1],trainable=True)

    def execute(self,inputs,convs,bias,soft_thresh,boundary_conditions='Circulant'):
        if len(inputs.shape)<4:
            inputs=tf.expand_dims(inputs,3)
        if self.conv_shape[1]>1 and inputs.shape[3]==1:
            inputs=tf.tile(inputs,(1,1,1,self.conv_shape[1]))
            inputs=inputs/tf.math.sqrt(tf.constant(self.conv_shape[1],dtype=tf.float32))
        if boundary_conditions=='Circulant':
            inputs_long=tf.concat([inputs[:,(-self.filter_length):,:],inputs,inputs[:,0:self.filter_length,:,:]],1)
            inputs_long=tf.concat([inputs_long[:,:,(-self.filter_length):,:],inputs_long,inputs_long[:,:,0:self.filter_length,:]],2)
        elif boundary_conditions=='Neumann':
            inputs_long=tf.concat([tf.reverse(inputs[:,:self.filter_length,:],[1]),inputs,tf.reverse(inputs[:,(-self.filter_length):,:,:],[1])],1)
            inputs_long=tf.concat([tf.reverse(inputs_long[:,:,:self.filter_length,:],[2]),inputs_long,tf.reverse(inputs_long[:,:,(-self.filter_length):,:],[2])],2)
        else:
            raise ValueError('Unknown boundary conditions!')
        kernel=tf.transpose(convs,[2,3,1,0])
        x=tf.nn.conv2d(inputs_long,kernel,1,'VALID')
        x = tf.add(x, bias)
        if self.soft_shrinkage_activation:
            x=tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-soft_thresh,0))
        else:
            x=self.activation(x)
         
        if self.transposed_mat:
            if boundary_conditions=='Circulant':
                x_long=tf.concat([x[:,(-self.filter_length):,:,:],x,x[:,0:self.filter_length,:,:]],1)
                x_long=tf.concat([x_long[:,:,(-self.filter_length):,:],x_long,x_long[:,:,0:self.filter_length,:]],2)
            elif boundary_conditions=='Neumann':
                x_long=tf.concat([tf.reverse(x[:,:self.filter_length:,:,:],[1]),x,tf.reverse(x[:,(-self.filter_length):,:,:],[1])],1)
                x_long=tf.concat([tf.reverse(x_long[:,:,:self.filter_length,:],[2]),x_long,tf.reverse(x_long[:,:,(-self.filter_length):,:],[2])],2)
            else:
                raise ValueError('Unknown boundary conditions!')
            kernels=tf.reverse(convs,[2])
            kernels=tf.reverse(kernels,[3])
            kernels=tf.transpose(kernels,[1,0,2,3])
            kernel=tf.transpose(kernels,[2,3,1,0])
            outputs=tf.nn.conv2d(x_long,kernel,1,'VALID')
        else:
            outputs=x
        if outputs.shape[3]==1:
            #if outputs.shape[0] is None:
            #    return outputs
            outputs=tf.reduce_sum(outputs,3)
            #outputs=tf.reshape(outputs,(outputs.shape[0],outputs.shape[1]))
        return outputs
    
    def call(self, inputs):
        return self.execute(inputs,self.convs,self.bias,self.soft_thresh)



class ScaleLayer(tf.keras.layers.Layer):
    # This layer scales the data points by a factor. If the factor in the constructor is given explicitly, then
    # the factor is not learnable. If factor in the constructor is None, the factor will be learned.
    # Input of the constructor: factor.
    def __init__(self,factor=None):
        super(ScaleLayer,self).__init__()
        self.fix_factor=factor
    def build(self,input_shape):
        if self.fix_factor is None:
            self.factor=self.add_weight("factor",initializer=tf.constant_initializer(1.),shape=[1],trainable=True)
    def call(self,inputs):
        if self.fix_factor is None:
            return self.factor*inputs
        return self.fix_factor*inputs
