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
from scipy.io import loadmat
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import sys
import os
import time
import math
from core.layers import *
import pickle


class StiefelModel(Model):
    # Implements a neural network consisting out of Stiefel Layers
    # Inputs:   dHidden         = list of the numbers of neurons in the hidden layers. The lenght of dHidden 
    #                             specifies the number of layers. Set it to [None]*n_layers for the convolutional 
    #                             case.
    #           lambdas         = list of thresholds for the soft shrinkage activation in the layers.
    #                             if lambdas is a float every Layer will be generated with threshold lambdas.
    #                             For other activation functions set lambda to None.
    #           activation      = activation function for the Stiefel Layers. None for soft shrinkage
    #           transposed_mat  = if false take layers of the form sigma(Ax+b) instead of A^T sigma(Ax+b).
    #                             Default: True.
    #           lastLayer       = True for appending an unconstrained Layer at the end (e.g. for softshrinkage)
    #                             otherwise, it should be set to None
    #           lastActivation  = If lastLayer is None, this has no effect. Otherwise this is the Activation function
    #                             for the unconstrained layer. None for softmax.
    #           convolutional   = list of booleans, whether the layers are convolutional.
    #           conv_shapes     = number of input and output filters. None for (1,1). 
    #           filter_length   = In the convolutional case: None for full filters. If filter_length is an integer
    #                             we use convolution filters of size 2*filter_length.
    #           dim             = Dimension of the data points (1 for signals, 2 for images etc.)
    #           fast_execution  = changes the execution order for a faster execution. Just applicable for fully
    #                             convolutional networks
    #           scale_layer     = Multiplicates the outputs with a scalar. Set it to False for deactivating the
    #                             scaling layer, True for learning the scalar and to a float for a fixed scalar.
    def __init__(self,dHidden,lambdas,activation=None,lastLayer=None,lastActivation=None,transposed_mat=True, convolutional=False,conv_shapes=None,filter_length=None,dim=1,fast_execution=False,scale_layer=False):
        super(StiefelModel,self).__init__()
        self.num_layers=len(dHidden)
        if type(conv_shapes)!=list:
            conv_shapes=[conv_shapes for i in range(self.num_layers)]
        if type(filter_length)!=list:   
            filter_length=[filter_length for i in range(self.num_layers)]
        if type(lambdas)!=list:
            #lambdas=lambdas*np.ones(self.num_layers)
            lambdas=self.num_layers*[lambdas]
        if type(convolutional)!=list:
            convolutional=[convolutional for i in range(0,self.num_layers)]
        if type(transposed_mat)!=list:
            transposed_mat=[transposed_mat for i in range(0,self.num_layers+1)]
        if type(activation)!=list:
            activation=[activation for i in range(0,self.num_layers)]
        self.stiefel=[]
        self.dim=dim
        self.fast_execution=fast_execution
        for i in range(0,self.num_layers):
            if convolutional[i]:
                if conv_shapes[i] is None:
                    conv_shapes[i]=(1,1)
                if filter_length[i] is None:
                    if dim==1:
                        self.stiefel.append(StiefelConv1D_full(dHidden[i],conv_shape=conv_shapes[i],soft_thresh=lambdas[i],activation=activation[i],transposed_mat=transposed_mat[i]))
                    else:
                        raise ValueError('Convolutional layers with full filter length are only implemented for dim=1')
                else:
                    if dim==1:
                        self.stiefel.append(StiefelConv1D(conv_shape=conv_shapes[i],soft_thresh=lambdas[i],activation=activation[i],transposed_mat=transposed_mat[i],filter_length=filter_length[i]))
                    elif dim==2:
                        self.stiefel.append(StiefelConv2D(conv_shape=conv_shapes[i],soft_thresh=lambdas[i],activation=activation[i],transposed_mat=transposed_mat[i],filter_length=filter_length[i]))
                    else:
                        raise ValueError('Convolutional layers with sparse filter length are only implemented for dim=1')
            else:
                if dim==1:
                    self.stiefel.append(StiefelDense1D(dHidden[i],lambdas[i],activation=activation[i],transposed_mat=transposed_mat))
                elif dim==2:
                    self.stiefel.append(StiefelDense2D(dHidden[i],lambdas[i],activation=activation[i],transposed_mat=transposed_mat))
                else:
                    raise ValueError('Dense layers are only implemented for dim=1 and dim=2')
        if lastLayer is None:
            self.lastLayer=None
        else:
            if lastActivation is None:
                self.lastLayer=Dense(lastLayer,activation='softmax',use_bias=False)
            else:
                self.lastLayer=Dense(lastLayer,activation=lastActivation,use_bias=False)
        if not type(scale_layer)==bool:
            self.scale=ScaleLayer(scale_layer)
        elif scale_layer:
            self.scale=ScaleLayer()
        else:
            self.scale=None
                

    def call(self,x,batch_size=None):
        if batch_size is None:
            if self.fast_execution:
                full=not self.stiefel[0].limited_filter
                mat=tf.reduce_sum(self.stiefel[0].convs,1)
                mat=tf.expand_dims(mat,1)
                mat/=tf.math.sqrt(tf.constant(self.stiefel[0].convs.shape[1],tf.float32))
                x=apply_convs(mat,x,full=full)
                x=tf.add(x,self.stiefel[0].bias)
                x=self.stiefel[0].activation(x)
                for i in range(1,len(self.stiefel)):
                    convs_old=tf.identity(self.stiefel[i-1].convs)
                    convs_old=transpose_convs(convs_old,full=full)
                    mat=conv_mult(self.stiefel[i].convs,convs_old,full=full)
                    x=apply_convs(mat,x,full=full)
                    x=tf.add(x,self.stiefel[i].bias)
                    x=self.stiefel[i].activation(x)
                convs_old=tf.identity(self.stiefel[-1].convs)
                convs_old=transpose_convs(convs_old,full=full)
                mat=tf.reduce_sum(convs_old,0)
                mat=tf.expand_dims(mat,0)
                mat/=tf.math.sqrt(tf.constant(convs_old.shape[0],tf.float32))
                x=apply_convs(mat,x,full=full)
                x=tf.reduce_sum(x,-1)
                if not self.scale is None:
                    x=self.scale(x)
                return x
            else:  
                for i in range(0,self.num_layers):
                    x=self.stiefel[i](x)
                if not (self.lastLayer is None):
                    if self.dim==1:
                        x=self.lastLayer(x)
                if self.dim==2 and len(x.shape)==4:
                    fac=tf.math.sqrt(tf.constant(x.shape[3],dtype=tf.float32))
                    x=tf.reduce_sum(x,axis=3)
                    x/=fac
                    if not self.lastLayer is None:
                        x_shape=x.shape[1:]
                        x=tf.reshape(x,[-1,tf.reduce_prod(x.shape[1:])])
                        x=self.lastLayer(x)
                        x=tf.reshape(x,[-1,x_shape[0],x_shape[1]])
                if self.dim==1 and len(x.shape)==3:
                    fac=tf.math.sqrt(tf.constant(x.shape[2],dtype=tf.float32))
                    x=tf.reduce_sum(x,axis=2)
                    x/=fac
                    if not self.lastLayer is None:
                        raise ValueError('Not implemented')
                if not self.scale is None:
                    x=self.scale(x)
                return x
        else:
            my_ds=tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
            out=[]
            for batch in my_ds:
                preds=self(batch)
                out.append(preds)
            out=tf.concat(out,0)
            return out
    

def loadData(fileName,num_data):
    # Loads training and test data from a .mat file
    # Inputs:   fileName    = Path from the parent directory to the file with '.mat'
    #           num_data    = number of training samples
    # Outputs:  Data: x_train, y_train, x_test, y_test
    x=loadmat('data/'+fileName)['signals']
    x_train=x[:,0:num_data].transpose()
    y_train=x_train+0.1*tf.random.normal(shape=(num_data,128))
    x_test=x[:,-1000:].transpose()
    y_test=x_test+0.1*tf.random.normal(shape=(1000,128))
    return x_train,y_train,x_test,y_test

def meanPSNR(predictions,ground_truth,one_dist=False):
    # Computes the mean psnr of data of the form [:,...].
    # The definition of the PSNR can be found e.g. in [2].
    # Inputs:
    #       predictions  - Array of shape [:,...]
    #       ground_truth - Array of shape [:,...]
    #       one_dist     - Set the maximal intensity to 1 (e.g. for images)
    # Outputs:
    #       PSNR
    if len(predictions.shape)==1:
        predictions=np.array([predictions])
    if len(ground_truth.shape)==1:
        ground_truth=np.array([ground_truth])
    MSE=np.sum((predictions-ground_truth)*(predictions-ground_truth),axis=1)
    while len(MSE.shape)>1:
        MSE=np.sum(MSE,axis=1)
    MSE/=np.prod(predictions.shape[1:])
    if one_dist:
        exp_psnr=1./MSE
    else:
        exp_psnr=(np.max(ground_truth,axis=1)-np.min(ground_truth,axis=1))**2/MSE
    psnr=10*np.log10(exp_psnr)
    return np.sum(psnr)/predictions.shape[0]

def MSE(pred,truth):
    # Computes the mean of the L2 error along the first axis
    # Inputs:
    #      pred   - predictions
    #      truth  - ground truth 
    # Outputs:
    #      result
    if len(pred.shape)==2:
        return np.sum((pred-truth)*(pred-truth))/(pred.shape[0]*pred.shape[1])
    if len(pred.shape)==1:
        return np.sum((pred-truth)*(pred-truth))/len(pred)
    return -1

def plotSaveSignal(signal,fileName,limits=None):
    # easy function to plot a signal.
    fig=plt.figure()
    plt.plot(signal,c='black')
    if not limits is None:
        plt.ylim(limits)
    fig.savefig(fileName,dpi=1200)
    plt.close(fig)

def train(model,x_train,y_train,x_test,y_test,EPOCHS=5,learn_rate=1,batch_size=32,loss_type='MSE',show_accuracy=None,progress_bar=True,residual=False):
    # Implements SGD on the Stiefel manifold for minimizing the loss of a StiefelModel.
    # Inputs:   model       = StiefelModel which will be trained
    #           Data:       x_train, y_train, x_test, y_test
    #           EPOCHS      = number of training epochs, default: 5
    #           learn_rate  = learning rate, default: 1
    #           batch size  = batch size, default: 32
    #           loss_type   = type of loss function. Default: MSE. 
    #                           'crossEntropy' for CategorialCrossentropy
    num_data=x_train.shape[0]
    train_accuracy=None
    test_accuracy=None
    if loss_type=='MSE':
        loss_object=tf.keras.losses.MeanSquaredError()
        if show_accuracy is None:
            show_accuracy=False
    elif loss_type=='crossEntropy':
        loss_object=tf.keras.losses.CategoricalCrossentropy()
        if show_accuracy is None:
            show_accuracy=True
    elif callable(loss_type):
        loss_object=loss_type
        if show_accuracy is None:
            show_accuracy=False
    else: 
        print('Loss function unknown. Take MSE-Loss')
        loss_object=tf.keras.losses.MeanSquaredError()
        if show_accuracy is None:
            show_accuracy=False
    if show_accuracy:
        train_accuracy=tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        test_accuracy=tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    train_loss=tf.keras.metrics.Mean(name="train_loss")
    test_loss=tf.keras.metrics.Mean(name="test_loss")
    test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)

    @tf.function
    def train_step_Cayley(inputs, outputs,step):
        factor=1
        s=2
        with tf.GradientTape() as tape:
            predictions=model(inputs)
            loss=loss_object(outputs,predictions)
        gradients=tape.gradient(loss, model.trainable_variables)
        num_layers=len(model.stiefel)
        var_num=0
        soft_thresh_sum=0
        soft_thresh_layers=0
        for i in range(0,num_layers):
            model.stiefel[i].bias.assign_sub(factor*learn_rate*gradients[var_num+1])
            if model.stiefel[i].conv:
                ### Convolutional Layer
                smaller=gradients[var_num].shape[0]<gradients[var_num].shape[1]
                convs=model.stiefel[i].convs
                if smaller:
                    convs=transpose_convs_full_filter(convs)
                    gradients[var_num]=transpose_convs_full_filter(gradients[var_num])
                W_hat=conv_mult_full_filter(gradients[var_num],transpose_convs_full_filter(convs))-.5*conv_mult_full_filter(convs,conv_mult_full_filter(transpose_convs_full_filter(convs),conv_mult_full_filter(gradients[var_num],transpose_convs_full_filter(convs))))
                W=W_hat-transpose_convs_full_filter(W_hat)
                new_convs=tf.identity(convs)
                for ind in range(1,s+1):
                    new_convs=convs-(factor*learn_rate)/2*conv_mult_full_filter(W,convs+new_convs)
                if smaller:
                    model.stiefel[i].convs.assign(transpose_convs_full_filter(new_convs))
                else:
                    model.stiefel[i].convs.assign(new_convs)
            else:
                # Dense Layer
                smaller=gradients[var_num].shape[0]<gradients[var_num].shape[1]
                matrix=model.stiefel[i].matrix
                if smaller:
                    gradients[var_num]=tf.transpose(gradients[var_num])
                    matrix=tf.transpose(matrix)
                W_hat=tf.linalg.matmul(gradients[var_num],tf.transpose(matrix))-0.5*tf.linalg.matmul(matrix,tf.linalg.matmul(tf.transpose(matrix),tf.linalg.matmul(gradients[var_num],tf.transpose(matrix))))
                W=W_hat-tf.transpose(W_hat)
                new_mat=matrix
                for ind in range(1,s+1):
                    new_mat=matrix-(factor*learn_rate)/2*tf.matmul(W,matrix+new_mat)
                if smaller:
                    model.stiefel[i].matrix.assign(tf.transpose(new_mat))
                else:
                    model.stiefel[i].matrix.assign(new_mat)
            if model.stiefel[i].learn_soft_thresh:
                soft_thresh_layers+=1
                model.stiefel[i].soft_thresh.assign_sub(learn_rate*gradients[var_num+2])
                soft_thresh_sum=tf.exp(model.stiefel[i].soft_thresh)+soft_thresh_sum
                var_num+=1
            var_num+=2
        if soft_thresh_layers>0:
            new_soft_thresh=tf.math.log(soft_thresh_sum/soft_thresh_layers)
            for i in range(len(model.stiefel)):
                if model.stiefel[i].learn_soft_thresh:
                    model.stiefel[i].soft_thresh.assign(new_soft_thresh)
        if not (model.lastLayer is None):
            if model.lastLayer.use_bias:
                model.lastLayer.bias.assign_sub(learn_rate*gradients[var_num+1])
                model.lastLayer.kernel.assign_sub(learn_rate*gradients[var_num])
                var_num+=2
            else:
                model.lastLayer.kernel.assign_sub(learn_rate*gradients[var_num])
                var_num+=1
        if not model.scale is None:
            if hasattr(model.scale, 'factor'):
                model.scale.factor.assign_sub(learn_rate*gradients[var_num]/100.)
                model.scale.factor.assign(tf.math.minimum(tf.math.maximum(model.scale.factor,0.1),1.9))
                var_num+=1
        if not (train_accuracy is None):
            train_accuracy(outputs,predictions)
        train_loss(loss)

    #@tf.function
    def test_step(inputs, outputs):
        predictions=model(inputs)
        t_loss=loss_object(outputs,predictions)
        if not (test_accuracy is None):
            test_accuracy(outputs,predictions)
        test_loss(t_loss)

    test_loss_vals=[] 
    train_loss_vals=[]
    psnrs=[]
    for test_inputs,test_outputs in test_ds:
        test_step(test_inputs,test_outputs)
    test_loss_vals.append(test_loss.result())
    print('Initial: Test Loss: ' + str(float(test_loss.result())), end=', ')
    if not (test_accuracy is None):
        print('Test Accuracy : '+str(float(test_accuracy.result())),end=', ')
        test_accuracy.reset_states()
    test_loss.reset_states()
    pred=model(x_test)
    if pred.shape[1]>y_test.shape[1]:
        pred=1/tf.math.sqrt(2.) * (pred[:,:pred.shape[1]//2]+pred[:,pred.shape[1]//2:])
    err=np.sum(((pred-y_test)*(pred-y_test)).numpy())/len(x_test)
    if residual:
        psnr_test=meanPSNR(x_test-pred,x_test-y_test)
    else:
        psnr_test=meanPSNR(pred,y_test)
    psnrs.append(psnr_test)
    print('MSE: ' +str(err)+' PSNR: '+str(psnr_test))
    myfile=open("log.txt","w")
    myfile.write("Log file for training\n")
    myfile.close()
    step=0
    myweights=model.trainable_weights
    if not (test_accuracy is None):
        max_accuracy=0;
    times=[0.]
    for epoch in range(EPOCHS):
        train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(num_data).batch(batch_size)
        count=0
        i=0
        tic=time.time()
        anz_steps=int(num_data/batch_size)
        print('Epoch '+str(epoch+1)+':')
        for inputs,outputs in train_ds:
            while count==round(i*anz_steps / 40.0) and i<40 and progress_bar:
                mytime=round(time.time()-tic)
                if i==0:
                    left=-1
                else:
                    left=round((40-i)*mytime/(i)) 
                sys.stdout.write("\r[{0}>{1}]      Step {2} / {3}      Time: {4} s, left: {5} s    ".format("="*i,"_"*(39-i),str(count),str(anz_steps),str(mytime),str(left)))
                sys.stdout.flush()
                i=i+1
            count+=1
            step+=1
            train_step_Cayley(inputs,outputs,tf.constant(float(epoch)))
        mytime=round(time.time()-tic)
        my_time=time.time()-tic
        times.append(times[-1]+my_time)
        if progress_bar:
            sys.stdout.write("\r[{0}]      Step {1} / {1}      Time: {2} s, left: 0 s     \n".format("="*40,str(anz_steps),str(mytime)))
            sys.stdout.flush()
        else:
            print('Time: {0:2.2f}'.format(times[-1]))
        for test_inputs,test_outputs in test_ds:
            test_step(test_inputs,test_outputs)
        myfile=open("log.txt","a")
        myfile.write('\nEpoch '+str(epoch+1)+':\n')
        myfile.write('Loss: '+str(float(train_loss.result()))+', Test Loss: '+str(float(test_loss.result())))
        print('Loss: {0:2.6f}, Test Loss: {1:2.6f}'.format(train_loss.result(),test_loss.result()),end=', ')
        test_loss_vals.append(test_loss.result())
        train_loss_vals.append(train_loss.result())
        train_loss.reset_states()
        test_loss.reset_states()
        nan_appeared=False
        for arr in model.trainable_weights:
            if np.isnan(arr.numpy()).any():
                nan_appeared=True
        if nan_appeared:
            print('\nNaN appeared!')
            break
        else:
            myweights=model.trainable_weights
        if not (train_accuracy is None):
            print('Train Accuracy: {0:2.4f}'.format(train_accuracy.result()),end=', ')
            myfile.write('\nTrain Accuracy: '+str(float(train_accuracy.result())))
            train_accuracy.reset_states()
        if not (test_accuracy is None):
            print('Test Accuracy: {0:2.4f}'.format(test_accuracy.result()),end=', ')
            myfile.write('\nTest Accuracy: '+str(float(test_accuracy.result())))
            if float(test_accuracy.result())>max_accuracy:
                max_accuracy=float(test_accuracy.result())
            print('Maximal Test Accuracy: {0:2.4f}'.format(max_accuracy),end=', ')
            myfile.write('\nMaximal Test Accuracy: '+str(max_accuracy))
            test_accuracy.reset_states()
        pred=model(x_test)
        if pred.shape[1]>y_test.shape[1]:
            pred=1/tf.math.sqrt(2.) * (pred[:,:pred.shape[1]//2]+pred[:,pred.shape[1]//2:])
        err=np.sum(((pred-y_test)*(pred-y_test)).numpy())/len(x_test)
        if residual:
            psnr_test=meanPSNR(x_test-pred,x_test-y_test)
        else:
            psnr_test=meanPSNR(pred,y_test)
        psnrs.append(psnr_test)
        print('MSE: {0:2.4f}, PSNR: {1:2.2f}'.format(err,psnr_test)) 
        myfile.write('\nMSE: ' +str(err)+' PSNR: '+str(psnr_test))
        myfile.close()
    return test_loss_vals,train_loss_vals,times,psnrs

def train_conv(model,epochs,train_ds,backup_dir,noise_level,penalty=1e4):
    # Trains a cPNN for denoising with orthogonality penalty term using the Adam optimizer
    # INPUTS:
    #       - model         - StiefelModel to be trained
    #       - epochs        - number of epochs for training
    #       - train_ds      - tensorflow data set containing the training data
    #       - backup_dir    - directory to save the weights
    #       - noise_level   - gaussian noise with standard deviation noise_level is added to get the corrupted data
    #       - penalty       - optional. Weight of the orthogonality penalty term. Default: 1e4
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    # declare penalty term to ensure orthogonality
    def orth_penalty(model):
        trainable_weights_counter=0
        beta=penalty
        out=0.
        for i in range(len(model.stiefel)):
            convs=model.stiefel[i].convs
            smaller=model.stiefel[i].conv_shape[0]<model.stiefel[i].conv_shape[1]            
            convs_trans=tf.transpose(convs,[1,0,2,3])
            convs_trans=tf.reverse(convs_trans,[2])
            convs_trans=tf.reverse(convs_trans,[3])
            if smaller:
                temp=convs
                convs=convs_trans
                convs_trans=temp
            filter_length=model.stiefel[i].filter_length
            iden=np.zeros((convs.shape[1],convs.shape[1],4*filter_length+1,4*filter_length+1),dtype=np.float32)            
            for j in range(convs.shape[1]):
                iden[j,j,2*filter_length,2*filter_length]=1
            iden=tf.constant(iden)
            C_r=tf.reverse(convs,[2])
            C_r=tf.reverse(C_r,[3])
            C_inp=tf.transpose(C_r,[1,2,3,0])
            C_inp_long=tf.concat([tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2],C_inp.shape[3])),C_inp,tf.zeros((C_inp.shape[0],filter_length,C_inp.shape[2],C_inp.shape[3]))],1)
            C_inp_long=tf.concat([tf.zeros((C_inp_long.shape[0],C_inp_long.shape[1],filter_length,C_inp_long.shape[3])),C_inp_long,tf.zeros((C_inp_long.shape[0],C_inp_long.shape[1],filter_length,C_inp_long.shape[3]))],2)
            C_trans_kernel=tf.transpose(convs_trans,[2,3,1,0])
            C_T_C_out=tf.nn.conv2d(C_inp_long,C_trans_kernel,1,'SAME')
            C_T_C=tf.transpose(C_T_C_out,[3,0,1,2])
            out+=tf.reduce_sum((C_T_C-iden)**2)      
            trainable_weights_counter+=2
        return beta*out

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            noise=noise_level*tf.random.normal(tf.shape(batch))
            preds=model(batch+noise)
            val1=tf.reduce_sum((preds-noise)**2)
            val2=orth_penalty(model)
            val=val1+val2
        grad=tape.gradient(val,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        return val1,val2

    @tf.function
    def test_step(batch):
        noise=noise_level*tf.random.normal(tf.shape(batch))
        preds=model(batch+noise)
        err=tf.reduce_sum((preds-noise)**2)
        return err

    obj=0.
    if not os.path.isdir(backup_dir):
        os.mkdir(backup_dir)
    for batch in train_ds:
        obj+=test_step(batch)
    print(obj.numpy())   
    print(orth_penalty(model).numpy())
    myfile=open(backup_dir+"/adam_log.txt","w")
    myfile.write("Log file for training\n")
    myfile.write(str(obj.numpy()))
    myfile.write("\n")
    myfile.write(str(orth_penalty(model).numpy()))
    myfile.write("\n")
    myfile.close()
    for ep in range(epochs):
        counter=0
        obj=0.
        for batch in train_ds:
            counter+=1
            val,val2=train_step(batch)
            obj+=val
        print('Training Error: ' + str(obj.numpy()))
        print(orth_penalty(model).numpy())
        myfile=open(backup_dir+"/adam_log.txt","a")
        myfile.write('Training Error: ' + str(obj.numpy())+'\n')
        myfile.write(str(orth_penalty(model).numpy())+'\n')
        myfile.close()
        obj=0.
                
        # save results
        with open(backup_dir+'/adam.pickle','wb') as f:
            pickle.dump(model.trainable_variables,f)
        if ep%5==0:
            with open(backup_dir+'/adam'+str(ep+1)+'.pickle','wb') as f:
                pickle.dump(model.trainable_variables,f)
