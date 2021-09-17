# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.
#
# Please cite the paper if you use this code.
#
# It implements a convolutional PNN for denoising as in the paper.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import conv_denoiser.train_conv_denoiser_BSDS
import conv_denoiser.train_conv_denoiser_BSDS_free
import conv_denoiser.test_conv_denoiser_BSD68
scales=[1.,1.99,5.,10.]
pretrained_weights=True
# Train and test scaled cPNNs
for scale in scales:
    print('Scale ' + str(scale) + ':')
    if not pretrained_weights:
        conv_denoiser.train_conv_denoiser_BSDS.run(scale=scale,noise_level=25./255.)
    else:
        print('Using pretrained weights.')
    conv_denoiser.test_conv_denoiser_BSD68.run(scale=scale,noise_level=25./255.,pretrained_weights=pretrained_weights)
# Train and test an unconstrained CNN for comparison
print('Unconstrained:')
if not pretrained_weights:
    conv_denoiser.train_conv_denoiser_BSDS_free.run(noise_level=25./255.)
else:
    print('Using pretrained weights.')
conv_denoiser.test_conv_denoiser_BSD68.run(scale=None,noise_level=25./255.,pretrained_weights=pretrained_weights)

