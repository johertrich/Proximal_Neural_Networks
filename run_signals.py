# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
# The script reproduces the numerical example with the signals in the paper.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Train all the networks, which are used in the paper
import signals_example.PNN_signals_all
print('Train all PNNs: ')
signals_example.PNN_signals_all.run()

# Learn the parameter of the threshold in the soft shrinkage via gradient descent
import signals_example.haar_learn_lambda
print('Learning threshold in softshrinkage (gradient descent): ')
signals_example.haar_learn_lambda.run()

# Learn the parameter of the threshold in the soft shrinkage via cross validation
import signals_example.haar_cv
print('Learning threshold in softshrinkage (cross validation): ')
signals_example.haar_cv.run()

# generate the plots
import signals_example.gen_plots
print('Generate final result plots: ')
signals_example.gen_plots.run()

