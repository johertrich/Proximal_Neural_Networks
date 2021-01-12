# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# arXiv Preprint#2011.02281, 2020.
#
# Please cite the paper if you use this code.
#
# In this file we implement a PnP-Deblurring for the BSD68 data set.
#
from core.stiefel_network import *
import os
import numpy as np
import numpy.random
from PIL import Image
from conv_denoiser.readBSD import *
import bm3d
from scipy.io import loadmat,savemat
from conv_denoiser.PnP_algs import *
from scipy.interpolate import griddata
from pyunlocbox import functions,solvers
import pickle

def run(scale=1.99,sigma_blur=0.1,noise_level_denoiser=0.005,num=None,method='FBS',pretrained_weights=True):
    # declare model
    act=tf.keras.activations.relu
    num_filters=64
    max_dim=128
    num_layers=8
    sizes=[None]*(num_layers)
    conv_shapes=[(num_filters,max_dim)]*num_layers
    filter_length=5
    model=StiefelModel(sizes,None,convolutional=True,filter_length=filter_length,dim=2,conv_shapes=conv_shapes,activation=act,scale_layer=scale)
    pred=model(tf.random.normal((10,40,40)))
    model.fast_execution=True

    # load weights    
    if pretrained_weights:
        file_name='data/pretrained_weights/scale'+str(scale)+'_noise_level'+str(noise_level_denoiser)+'.pickle'
    else:
        if num is None:
            file_name='results_conv/scale'+str(scale)+'_noise_level'+str(noise_level_denoiser)+'/adam.pickle'
        else:
            file_name='results_conv/scale'+str(scale)+'_noise_level'+str(noise_level_denoiser)+'/adam'+str(num)+'.pickle'
    with open(file_name,'rb') as f:
        trainable_vars=pickle.load(f)
    for i in range(len(model.trainable_variables)):
        model.trainable_variables[i].assign(trainable_vars[i])
    beta=1e8
    project=True
    if project:
        # project convolution matrices on the Stiefel manifold
        for i in range(len(model.stiefel)):
            convs=model.stiefel[i].convs
            smaller=convs.shape[0]<convs.shape[1]
            if smaller:
                convs=transpose_convs(convs)
            iden=np.zeros((convs.shape[1],convs.shape[1],4*filter_length+1,4*filter_length+1),dtype=np.float32)            
            for j in range(convs.shape[1]):
                iden[j,j,2*filter_length,2*filter_length]=1
            iden=tf.constant(iden)
            C=tf.identity(convs)
            def projection_objective(C):
                return 0.5*beta*tf.reduce_sum((conv_mult(transpose_convs(C),C)-iden)**2)+.5*tf.reduce_sum((C-convs)**2)
            for iteration in range(100):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(C)
                    val=projection_objective(C)
                    grad=tape.gradient(val,C)
                    grad_sum=tf.reduce_sum(grad*grad)
                hess=tape.gradient(grad_sum,C)
                hess*=0.5/tf.sqrt(grad_sum)
                C-=grad/tf.sqrt(tf.reduce_sum(hess*hess))
            if smaller:
                C=transpose_convs(C)
            model.stiefel[i].convs.assign(C)

    # load data
    test_directory='data/BSD68'
    fileList=os.listdir(test_directory+'/')
    fileList.sort()
    img_names=fileList
    save_path='results_conv/PnP_blur_'+method+str(sigma_blur)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'/blurred_data'):
        os.mkdir(save_path+'/blurred_data')
    if not os.path.isdir(save_path+'/l2tv'):
        os.mkdir(save_path+'/l2tv')
    psnr_sum=0.
    psnr_noisy_sum=0.
    psnr_l2tv_sum=0.
    error_sum=0.
    error_bm3d_sum=0.
    counter=0
    sig=sigma_blur
    sig_sq=sig**2
    noise_level=0.01
    kernel_width=9
    x_range=1.*np.array(range(kernel_width))
    kernel_x=np.tile(x_range[:,np.newaxis],(1,kernel_width))-.5*(kernel_width-1)
    y_range=1.*np.array(range(kernel_width))
    kernel_y=np.tile(y_range[np.newaxis,:],(kernel_width,1))-.5*(kernel_width-1)
    kernel=np.exp(-(kernel_x**2+kernel_y**2)/(2*sig_sq))
    kernel/=np.sum(kernel)
    kernel=tf.constant(kernel,dtype=tf.float32)
    myfile=open(save_path+"/psnrs.txt","w")
    myfile.write("PSNRs:\n")
    myfile.close()
    np.random.seed(25)
    for name in img_names:
        # load image and compute blurred version
        counter+=1
        img=Image.open(test_directory+'/'+name)
        img=img.convert('L')
        img_gray=1.0*np.array(img)
        img_gray/=255.0
        
        img_gray_pil=Image.fromarray(img_gray*255.0)
        img_gray_pil=img_gray_pil.convert('RGB')
        img_gray_pil.save(save_path+'/original'+name)  
        one_img=tf.ones(img_gray.shape)

        img_blurred=tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tf.constant(img_gray,dtype=tf.float32),0),-1),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'SAME')
        img_blurred=tf.squeeze(img_blurred).numpy()
        ones_blurred=tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tf.constant(one_img,dtype=tf.float32),0),-1),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'SAME')
        ones_blurred=tf.squeeze(ones_blurred).numpy()
        img_blurred/=ones_blurred
        noise=np.random.normal(0,1,img_blurred.shape)
        img_blurred+=noise_level*noise
        pad=kernel_width//2
        img_obs=img_blurred[pad:-pad,pad:-pad]
        img_start=np.pad(img_obs,((pad,pad),(pad,pad)),'edge')
        img_obs_big=np.concatenate([np.zeros((img_obs.shape[0],pad)),img_obs,np.zeros((img_obs.shape[0],pad))],1)
        img_obs_big=np.concatenate([np.zeros((pad,img_obs_big.shape[1])),img_obs_big,np.zeros((pad,img_obs_big.shape[1]))],0)
        savemat(save_path+'/blurred_data/'+name[:-4]+'_blurred.mat',{'img_blur': (img_blurred)*255})
        scalar=scale
        alpha_star=0.5
        conv_coord=1-scalar+2*alpha_star*scalar
        
        # declare functions for PnP
        def my_f(signal,inp_signal):
            signal_blurred=tf.nn.conv2d(tf.expand_dims(signal,-1),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'VALID')
            signal_blurred=tf.reshape(signal_blurred,signal_blurred.shape[:3])
            out=.5*tf.reduce_sum((signal_blurred-img_obs)**2)
            return out

        def prox_my_f(signal,lam,inp_signal):
            out_signal=tf.identity(signal)
            for i in range(50):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(out_signal)
                    term1=my_f(out_signal,inp_signal)
                    term2=.5*tf.reduce_sum((out_signal-signal)**2)
                    objective=term1/lam+term2
                    grad=tape.gradient(objective,out_signal)
                    grad_sum=tf.reduce_sum(grad**2)
                hess=.5*tape.gradient(grad_sum,out_signal)/tf.sqrt(grad_sum)
                out_signal-=grad/tf.sqrt(tf.reduce_sum(hess**2))
            return out_signal
                    
        def grad_f(signal):
            signal_blurred=tf.nn.conv2d(tf.expand_dims(signal,-1),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'SAME')
            signal_blurred_minus_inp=tf.reshape(signal_blurred,signal_blurred.shape[:3])-img_blurred

            AtA=tf.nn.conv2d(tf.expand_dims(signal_blurred_minus_inp,-1),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'SAME')
            AtA=tf.reshape(AtA,signal_blurred.shape[:3])
            return AtA
            

        #L2-TV
        def g(signal):
            signal_blurred=tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tf.constant(signal,tf.float32),-1),0),tf.expand_dims(tf.expand_dims(kernel,-1),-1),1,'VALID')
            signal_blurred=tf.squeeze(signal_blurred)
            signal_blurred=np.concatenate([np.zeros((signal_blurred.shape[0],pad)),signal_blurred.numpy(),np.zeros((signal_blurred.shape[0],pad))],1)
            signal_blurred=np.concatenate([np.zeros((pad,signal_blurred.shape[1])),signal_blurred,np.zeros((pad,signal_blurred.shape[1]))],0)
            return signal_blurred

        f1=functions.norm_tv(maxit=50,dim=2)
        l2tv_lambda=0.001
        f2=functions.norm_l2(y=img_obs_big,A=g,lambda_=1/l2tv_lambda)
        solver=solvers.forward_backward(step=0.5*l2tv_lambda)
        img_blurred2=tf.identity(img_start).numpy()
        l2tv=solvers.solve([f1,f2],img_blurred2,solver,maxit=100,verbosity='NONE')
        l2tv=l2tv['sol']

        def my_T(inp,model):
            my_fac=1.
            return (1-1/(conv_coord))*l2tv+1/(conv_coord)*(inp-model((inp-.5)*my_fac))

        # Compute PnP result
        if method=='FBS': 
            pred=PnP_FBS(model,l2tv[np.newaxis,:,:],tau=1.9,T_fun=my_T,eps=1e-3,fun=my_f)
        elif method=='ADMM':
            pred=PnP_ADMM(l2tv[np.newaxis,:,:],lambda x: my_T(x,model),gamma=.52,prox_fun=prox_my_f)
        else:
            raise ValueError('Unknown method!')
      
        # save results
        noisy=(img_start)*255
        reconstructed=(tf.reshape(pred,[pred.shape[1],pred.shape[2]]).numpy())*255.
        img_gray=(img_gray)*255.
        l2tv*=255
        error_sum+=tf.reduce_sum(((reconstructed-img_gray)/255.)**2).numpy()
        psnr=meanPSNR(tf.keras.backend.flatten(reconstructed[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,tf.keras.backend.flatten(img_gray[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,one_dist=True)
        psnr_l2tv=meanPSNR(tf.keras.backend.flatten(l2tv[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,tf.keras.backend.flatten(img_gray[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,one_dist=True)
        psnr_noisy=meanPSNR(tf.keras.backend.flatten(noisy[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,tf.keras.backend.flatten(img_gray[2*pad:-2*pad,2*pad:-2*pad]).numpy()/255.0,one_dist=True)
        print('PSNR of '+name+':                    '+str(psnr))
        print('PSNR L2TV of '+name+':               '+str(psnr_l2tv))
        print('PSNR of noisy '+name+':              '+str(psnr_noisy))
        psnr_sum+=psnr
        psnr_noisy_sum+=psnr_noisy
        psnr_l2tv_sum+=psnr_l2tv
        print('Mean PSNR PPNN:      '+str(psnr_sum/counter))
        print('Mean PSNR L2TV:      '+str(psnr_l2tv_sum/counter))
        print('Mean PSNR noisy:     '+str(psnr_noisy_sum/counter))
        myfile=open(save_path+"/psnrs.txt","a")
        myfile.write('PSNR of '+name+':                    '+str(psnr)+'\n')
        myfile.write('PSNR L2TV of '+name+':               '+str(psnr_l2tv)+'\n')
        myfile.write('PSNR of noisy '+name+':              '+str(psnr_noisy)+'\n')
        myfile.close()
        img=Image.fromarray(noisy)
        img=img.convert('RGB')
        img.save(save_path+'/noisy'+name)
        img=Image.fromarray(l2tv)
        img=img.convert('RGB')
        img.save(save_path+'/l2tv/l2tv'+name)
        img=Image.fromarray(reconstructed)
        img=img.convert('RGB')
        img.save(save_path+'/reconstructed'+name)       
    print('Mean PSNR on images: '+str(psnr_sum/len(img_names)))
    print('Mean PSNR on noisy images: '+str(psnr_noisy_sum/len(img_names)))

