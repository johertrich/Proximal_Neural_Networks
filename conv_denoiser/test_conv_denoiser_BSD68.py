# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# arXiv Preprint#2011.02281, 2020.
#
# Please cite the paper if you use this code.
#
from core.stiefel_network import *
from core.layers import *
import numpy as np
import numpy.random
from PIL import Image
from conv_denoiser.readBSD import *
import bm3d
import pickle
from scipy.io import savemat

def run(scale=1.99,noise_level=25./255.,num=None,pretrained_weights=True):
    # declare network
    act=tf.keras.activations.relu
    num_filters=64
    max_dim=128
    num_layers=8
    sizes=[None]*(num_layers)
    conv_shapes=[(num_filters,max_dim)]*num_layers
    filter_length=5
    if scale is None:
        model=StiefelModel(sizes,None,convolutional=True,filter_length=filter_length,dim=2,conv_shapes=conv_shapes,activation=act,scale_layer=False)
    else:
        model=StiefelModel(sizes,None,convolutional=True,filter_length=filter_length,dim=2,conv_shapes=conv_shapes,activation=act,scale_layer=scale)

    pred=model(tf.random.normal((10,40,40)))
    model.fast_execution=True

    if scale is None:
        # load weights
        if pretrained_weights:
            file_name='data/pretrained_weights/free_noise_level'+str(noise_level)+'.pickle'
        else:
            if num is None:
                file_name='results_conv/free_noise_level'+str(noise_level)+'/adam.pickle'
            else:
                file_name='results_conv/free_noise_level'+str(noise_level)+'/adam'+str(num)+'.pickle'
        with open(file_name,'rb') as f:
            trainable_vars=pickle.load(f)
        for i in range(len(model.trainable_variables)):
            model.trainable_variables[i].assign(trainable_vars[i])
    else:
        # load weights
        if pretrained_weights:
            file_name='data/pretrained_weights/scale'+str(scale)+'_noise_level'+str(noise_level)+'.pickle'
        else:
            if num is None:
                file_name='results_conv/scale'+str(scale)+'_noise_level'+str(noise_level)+'/adam.pickle'
            else:
                file_name='results_conv/scale'+str(scale)+'_noise_level'+str(noise_level)+'/adam'+str(num)+'.pickle'
        with open(file_name,'rb') as f:
            trainable_vars=pickle.load(f)
        for i in range(len(model.trainable_variables)):
            model.trainable_variables[i].assign(trainable_vars[i])
        beta=1e8
        project=True
        if project:
            # projection of the convolution matrices onto the Stiefel manifold
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
                

    np.set_printoptions(threshold=sys.maxsize)


    # set parameters
    test_directory='data/BSD68'
    fileList=os.listdir(test_directory+'/')
    fileList.sort()
    img_names=fileList
    sig=25.
    sig/=255.
    if scale is None:
        save_path='results_conv/denoise_results_free_noise_level'+str(noise_level)
    else:
        save_path='results_conv/denoise_results_scale'+str(scale)+'_noise_level'+str(noise_level)
    if not os.path.isdir('results_conv'):
        os.mkdir('results_conv')
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
    my_shift=10
    myfile=open(save_path+"/psnrs.txt","w")
    myfile.write("PSNRs:\n")
    myfile.close()
    np.random.seed(30)
    # Denoising
    for name in img_names:
        counter+=1
        img=Image.open(test_directory+'/'+name)
        img=img.convert('L')
        img_gray=1.0*np.array(img)
        img_gray/=255.0
        img_gray-=.5
        img_gray_noisy=img_gray+sig*np.random.normal(size=img_gray.shape)
        savemat(save_path+'/noisy_data/'+name[:-4]+'_noisy.mat',{'noisy': (img_gray_noisy+0.5)*255})
        img_gray_pil=Image.fromarray((img_gray+.5)*255.0)
        img_gray_pil=img_gray_pil.convert('RGB')
        img_gray_pil.save(save_path+'/original'+name)    
        pred=model(tf.expand_dims(tf.constant(img_gray_noisy,dtype=tf.float32),0))
        noisy=(img_gray_noisy+.5)*255
        reconstructed=((img_gray_noisy-np.reshape(pred.numpy(),img_gray.shape))+.5)*255.
        img_gray=(img_gray+.5)*255.
        res_bm3d=(bm3d.bm3d(img_gray_noisy,sig)+0.5)*255.
        error_sum+=tf.reduce_sum(((reconstructed-img_gray)/255.)**2).numpy()
        error_bm3d_sum+=tf.reduce_sum(((res_bm3d-img_gray)/255.)**2).numpy()
        psnr=meanPSNR(tf.keras.backend.flatten(reconstructed).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        psnr_noisy=meanPSNR(tf.keras.backend.flatten(noisy).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        psnr_bm3d=meanPSNR(tf.keras.backend.flatten(res_bm3d).numpy()/255.0,tf.keras.backend.flatten(img_gray).numpy()/255.0,one_dist=True)
        print('PSNR of '+name+':                    '+str(psnr))
        print('PSNR of BM3D '+name+':               '+str(psnr_bm3d))
        print('PSNR of noisy '+name+':              '+str(psnr_noisy))
        psnr_sum+=psnr
        psnr_noisy_sum+=psnr_noisy
        psnr_bm3d_sum+=psnr_bm3d
        print('Mean PSNR PPNN:                      '+str(psnr_sum/counter))
        print('Mean PSNR BM3D:                      '+str(psnr_bm3d_sum/counter))
        print('Mean Error PPNN:                     '+str(error_sum/counter))
        print('Mean Error BM3D:                     '+str(error_bm3d_sum/counter))
        print('Mean PSNR noisy:                     '+str(psnr_noisy_sum/counter))    
        myfile=open(save_path+"/psnrs.txt","a")
        myfile.write('PSNR of '+name+':                    '+str(psnr)+'\n')
        myfile.write('PSNR of BM3D '+name+':               '+str(psnr_bm3d)+'\n')
        myfile.write('PSNR of noisy '+name+':              '+str(psnr_noisy)+'\n')
        myfile.close()
        img=Image.fromarray(noisy)
        img=img.convert('RGB')
        img.save(save_path+'/noisy'+name)
        img=Image.fromarray(reconstructed)
        img=img.convert('RGB')
        img.save(save_path+'/reconstructed'+name)            
        img=Image.fromarray(res_bm3d)
        img=img.convert('RGB')
        img.save(save_path+'/other/BM3D'+name)            
    print('Mean PSNR on images: '+str(psnr_sum/len(img_names)))
    print('Mean PSNR on noisy images: '+str(psnr_noisy_sum/len(img_names)))

