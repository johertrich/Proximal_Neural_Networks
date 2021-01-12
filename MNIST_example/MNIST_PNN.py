# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
# In the following code, a PNN for MNIST classification is trained.
#
from core.stiefel_network import *
import numpy.matlib
import os

def run():
    if not os.path.exists('results_MNIST'):
        os.mkdir('results_MNIST')
    
    # load and preprocess data
    mnist=tf.keras.datasets.mnist

    (x_train,y_train),(x_test,y_test)=mnist.load_data()

    x_train=1.0*x_train
    x_test=1.0*x_test
    x_train_flat=[]
    x_test_flat=[]
    y_train_vec=[]
    y_test_vec=[]

    for i in range(0,len(x_train)):
        x_train_flat.append(x_train[i,:,:].reshape((28*28)))
        y_vec=np.zeros(10)
        y_vec[y_train[i]]=1.0
        y_train_vec.append(y_vec)

    for i in range(0,len(x_test)):
        x_test_flat.append(x_test[i,:,:].reshape((28*28)))
        y_vec=np.zeros(10)
        y_vec[y_test[i]]=1.0
        y_test_vec.append(y_vec)


    x_train=1.0*np.array(x_train_flat).astype(np.float32)
    y_train=1.0*np.array(y_train_vec).astype(np.float32)
    x_test=1.0*np.array(x_test_flat).astype(np.float32)
    y_test=1.0*np.array(y_test_vec).astype(np.float32)

    mean_x_train=1.0/len(x_train)*np.sum(x_train,axis=0)

    x_train=x_train-np.matlib.repmat(mean_x_train,len(x_train),1)
    x_test=x_test-np.matlib.repmat(mean_x_train,len(x_test),1)

    max_x_train=np.max(np.abs(x_train))
    x_train=x_train/max_x_train
    x_test=x_test/max_x_train

    # define model
    model=StiefelModel([784,784,400,400,200],0.5,lastLayer=10,activation=tf.keras.activations.relu,lastActivation='sigmoid')
    # train model    
    test_loss_vals,train_loss_vals,_,_=train(model,x_train,y_train,x_test,y_test,1000,learn_rate=5.0,loss_type='MSE',batch_size=1024,show_accuracy=True)
    os.rename('log.txt','results_MNIST/training_log.txt')
    model.save('results_MNIST/MNIST_PNN')
    # test model and generate outputs
    pred=model(x_test)
    pred_train=model(x_train)
    correct_train=0
    for i in range(0,len(x_train)):
        if np.argmax(pred_train[i,:])==np.argmax(y_train[i,:]):
            correct_train+=1
    print('Training set:')
    print('Correct classifications: '+str(correct_train)+' of '+str(len(x_train))+'. Accuracy: '+str(correct_train*1.0/len(x_train)))
    correct=0
    for i in range(0,len(x_test)):
        if np.argmax(pred[i,:])==np.argmax(y_test[i,:]):
            correct+=1
    print('Test set:')
    print('Correct classifications: '+str(correct)+' of '+str(len(x_test))+'. Accuracy: '+str(correct*1.0/len(x_test)))

    for i in range(0,len(test_loss_vals)):
        test_loss_vals[i]=test_loss_vals[i].numpy()
    for i in range(0,len(train_loss_vals)):
        train_loss_vals[i]=train_loss_vals[i].numpy()
    fig=plt.figure()
    plt.plot(train_loss_vals,c='blue')
    plt.plot(test_loss_vals,'--',c='orange')
    plt.yscale("log")
    fig.savefig('results_MNIST/MNIST_train_loss.png',dpi=1200)
    plt.close(fig)
