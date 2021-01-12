# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
# The script reproduces the numerical example with the MNIST classification in the paper.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Train PNN for MNIST classification
import MNIST_example.MNIST_PNN
print('Train PNN for classification: ')
MNIST_example.MNIST_PNN.run()

# Train comparison NN for MNIST classification
import MNIST_example.MNIST_comparison
print('Train comparison NN for classification: ')
MNIST_example.MNIST_comparison.run()

# Perform adversarial attacks
import MNIST_example.adversarial_MNIST
print('Perform adversarial attack: ')
MNIST_example.adversarial_MNIST.run()
