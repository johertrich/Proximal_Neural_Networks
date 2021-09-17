# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.
#
# Please cite the paper if you use this code.
#
# It implements a PnP iteration for denoising and deblurring as in the paper.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import conv_denoiser.train_conv_denoiser_BSDS
import conv_denoiser.BSD68_PnP_blur
import conv_denoiser.BSD68_PnP_noise

pretrained_weights=True

# Train scaled denoising cPNN if required
print('Denoising with scale 1.99:')
if not pretrained_weights:
    print('\n\nTRAIN PNN WITH SCALE FACTOR 1.99 AND NOISE LEVEL 25/255!\n\n')
    conv_denoiser.train_conv_denoiser_BSDS.run(scale=1.99,noise_level=25./255.)
else:
    print('Using pretrained weights.')
# run PnP for denoising
noise_levels_pnp=[0.075,0.1,0.125,0.15]
for nl in noise_levels_pnp:
    conv_denoiser.BSD68_PnP_noise.run(scale=1.99,noise_level_PnP=nl,noise_level_denoiser=25./255.,pretrained_weights=pretrained_weights)

# Train scaled denoising cPNN if required
print('Denoising with scale 5:')
if not pretrained_weights:
    print('\n\nTRAIN PNN WITH SCALE FACTOR 5 AND NOISE LEVEL 25/255!\n\n')
    conv_denoiser.train_conv_denoiser_BSDS.run(scale=5.,noise_level=25./255.)
else:
    print('Using pretrained weights.')
# run PnP for denoising
for nl in noise_levels_pnp:
    conv_denoiser.BSD68_PnP_noise.run(scale=5.,noise_level_PnP=nl,noise_level_denoiser=25./255.,pretrained_weights=pretrained_weights)

# Train scaled denoising cPNN if required
print('Deblurring with scale 1.99:')
if not pretrained_weights:
    print('\n\nTRAIN PNN WITH SCALE FACTOR 1.99 AND NOISE LEVEL 0.005!\n\n')
    conv_denoiser.train_conv_denoiser_BSDS.run(scale=1.99,noise_level=0.005)
else:
    print('Using pretrained weights.')
# run PnP for deblurring
blur_levels=[1.25,1.5,1.75,2.0]
for bl in blur_levels:
    print('Blur level ' + str(b1) +':')
    conv_denoiser.BSD68_PnP_blur.run(scale=1.99,sigma_blur=bl,noise_level_denoiser=0.005,method='FBS',pretrained_weights=pretrained_weights)
    conv_denoiser.BSD68_PnP_blur.run(scale=1.99,sigma_blur=bl,noise_level_denoiser=0.005,method='ADMM',pretrained_weights=pretrained_weights)

