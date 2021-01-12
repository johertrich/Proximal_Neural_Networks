# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# arXiv Preprint#2011.02281, 2020.
#
# Please cite the paper if you use this code.
#
# In this file we implement a PnP-Denoising for the BSD68 data set.
#
from core.stiefel_network import *
from core.layers import *
import numpy as np
import numpy.random
from PIL import Image
from conv_denoiser.readBSD import *
import bm3d
from scipy.io import savemat
from conv_denoiser.PnP_algs import *
import pickle

def run(scale=1.99,noise_level_PnP=0.1,noise_level_denoiser=25./255.,num=None,pretrained_weights=True):
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
    save_path='results_conv/PnP_FBS_noise'+str(noise_level_PnP)+'_scale'+str(scale)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'/noisy_data'):
        os.mkdir(save_path+'/noisy_data')
    if not os.path.isdir(save_path+'/other'):
        os.mkdir(save_path+'/other')
    psnr_sum=0.
    psnr_noisy_sum=0.
    psnr_bm3d_sum=0.
    error_sum=0.
    error_bm3d_sum=0.
    counter=0
    noise_level=noise_level_PnP
    np.random.seed(30)
    for name in img_names:
        # load image and add noise
        counter+=1
        img=Image.open(test_directory+'/'+name)
        img=img.convert('L')
        img_gray=1.0*np.array(img)
        img_gray/=255.0
        img_gray=img_gray[:-1,:-1]
        noise=np.random.normal(0,1,img_gray.shape)
        img_gray_noisy=img_gray+noise_level*noise

        img_gray_pil=Image.fromarray((img_gray)*255.0)
        img_gray_pil=img_gray_pil.convert('RGB')
        img_gray_pil.save(save_path+'/original'+name)    

        scalar=scale
        alpha_star=0.6
        conv_coord=1-scalar+2*alpha_star*scalar
        # compute BM3D-Result
        bm3d_result=bm3d.bm3d(img_gray_noisy,noise_level)
        def my_f(signal,inp_signal):
            return .5*tf.reduce_sum((signal-inp_signal)**2)

        def my_T(inp,model):
            return (1-1/(conv_coord))*bm3d_result+1/(conv_coord)*(inp-model(inp-.5))
        
        # Compute PnP
        if noise_level==0.15:
            pred=PnP_FBS(model,img_gray_noisy[np.newaxis,:,:],fun=my_f,tau=.58,T_fun=my_T,eps=1e-3) #sig 0.15
        elif noise_level==0.125:
            pred=PnP_FBS(model,img_gray_noisy[np.newaxis,:,:],fun=my_f,tau=.72,T_fun=my_T,eps=1e-3) #sig 0.125
        elif noise_level==0.1:
            pred=PnP_FBS(model,img_gray_noisy[np.newaxis,:,:],fun=my_f,tau=0.93,T_fun=my_T,eps=1e-3) #sig 0.1
        elif noise_level==0.075:
            pred=PnP_FBS(model,img_gray_noisy[np.newaxis,:,:],fun=my_f,tau=1.35,T_fun=my_T,eps=1e-3) #sig 0.075
        else:
            raise ValueError('Tau not fittet for this noise level!')
        # Save results
        noisy=(img_gray_noisy)*255.
        reconstructed=(tf.reshape(pred,[pred.shape[1],pred.shape[2]]).numpy())*255.
        bm3d_result*=255.
        img_gray=(img_gray)*255.
        error_sum+=tf.reduce_sum(((reconstructed-img_gray)/255.)**2).numpy()
        psnr=meanPSNR(tf.keras.backend.flatten(reconstructed).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        psnr_bm3d=meanPSNR(tf.keras.backend.flatten(bm3d_result).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        psnr_noisy=meanPSNR(tf.keras.backend.flatten(noisy).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        print('PSNR of '+name+':                    '+str(psnr))
        print('PSNR of bm3d '+name+':               '+str(psnr_bm3d))
        print('PSNR of noisy '+name+':              '+str(psnr_noisy))
        psnr_sum+=psnr
        psnr_noisy_sum+=psnr_noisy
        psnr_bm3d_sum+=psnr_bm3d
        print('Mean PSNR PPNN:      '+str(psnr_sum/counter))
        print('Mean PSNR BM3D:      '+str(psnr_bm3d_sum/counter))
        print('Mean Error PPNN:     '+str(error_sum/counter))
        print('Mean PSNR noisy:     '+str(psnr_noisy_sum/counter))
        img=Image.fromarray(noisy)
        img=img.convert('RGB')
        img.save(save_path+'/noisy'+name)
        img=Image.fromarray(reconstructed)
        img=img.convert('RGB')
        img.save(save_path+'/reconstructed'+name)          
        img=Image.fromarray(bm3d_result)
        img=img.convert('RGB')
        img.save(save_path+'/other/BM3D'+name)   
    print('Mean PSNR on images: '+str(psnr_sum/len(img_names)))
    print('Mean PSNR on noisy images: '+str(psnr_noisy_sum/len(img_names)))

