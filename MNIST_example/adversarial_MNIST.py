# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
# In the following code an adversarial attack is performed onto the PNN and the comparison NN.
#
from core.stiefel_network import *
import numpy.matlib


def run(model_name):
    # load and preprocess data
    model=tf.keras.models.load_model(model_name)
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

    # perform attack
    num_att=x_test.shape[0]
    failed=0
    already_wrong=0
    factors=[]
    norms=[]
    for i in range(num_att):
        x=tf.Variable(x_test[i:(i+1),:])
        out=model(x)
        y=np.argmax(out)
        with tf.GradientTape() as tape:
            classification=model(x)
            classification=classification[0,y]/tf.reduce_sum(classification)
        gradient=tape.gradient(classification,x)
        norm_gradient=norm_noise=np.sqrt(max_x_train*np.sum(gradient*gradient))    
        if np.argmax(out)==y:
            factor=1e-2
            my_class=y
            it=0
            while my_class==y and factor*norm_gradient<=1000:
                factor*=2            
                out=model(x-factor*gradient)
                my_class=np.argmax(out)
                it+=1
            if my_class==y:
                print('adversarial attack '+str(i)+' of '+str(num_att)+' failed')
                failed+=1
            else:
                norm_noise=factor*norm_gradient
                print('adversarial attack '+str(i)+' of '+str(num_att)+' worked with factor='+str(factor)+' and norm of noise '+str(norm_noise))
                factors.append(factor)
                norms.append(norm_noise)
        else:
            print('classification '+str(i)+' of '+str(num_att)+' already wrong')
            already_wrong+=1

    # write outputs
    mystring='Successful attacks: '+str(num_att-failed-already_wrong)+' Failed: '+str(failed)+' already wrong: '+str(already_wrong)+'\n'
    mystring=mystring+'Factors: Mean: ' +str(np.mean(np.array(factors)))+' Std: '+str(np.sqrt(np.var(np.array(factors))))+' Median: '+str(np.median(np.array(factors)))+'\n'
    mystring=mystring+'Norms: Mean: ' +str(np.mean(np.array(norms)))+' Std: '+str(np.sqrt(np.var(np.array(norms))))+' Median: ' +str(np.median(np.array(norms)))+'\n'
    myfile=open("results_MNIST/attacks.txt","a")
    myfile.write('Model: '+model_name+'\n')
    myfile.write(mystring)    
    myfile.close()
    print(mystring)

def run():
    run('results_MNIST/MNIST_PNN')
    run('results_MNIST/MNIST_comparison')
