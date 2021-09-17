# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.
#
# Please cite the paper if you use this code.
#
from core.stiefel_network import *
import numpy as np
import numpy.random
from conv_denoiser.readBSD import *
import pickle

def run(noise_level=25./255.):
    # load and preprocess data
    patch_size=40
    y_train,x_train=loadFromPath(patch_size=patch_size,shift=30,path='data/train_BSDS_png')
    y_test,x_test=loadFromPath(patch_size=patch_size,shift=40,path='data/test_BSDS_png')

    y_train=tf.cast(y_train,dtype=tf.float32)
    x_train=tf.cast(x_train,dtype=tf.float32)
    y_test=tf.cast(y_test,dtype=tf.float32)
    x_test=tf.cast(x_test,dtype=tf.float32)


    print(x_test.shape)
    print(x_train.shape)
    batch_size=30
    num_dat=int(np.floor(x_train.shape[0]*1.0/batch_size))*batch_size
    x_train=y_train+noise_level*tf.random.normal(y_train.shape)
    x_test=y_test+noise_level*tf.random.normal(y_test.shape)
    y_train=y_train[:num_dat]
    x_train=x_train[:num_dat]
    residual=True
    if residual:
        y_train=x_train-y_train
        y_test=x_test-y_test


    # declare network
    act=tf.keras.activations.relu
    num_filters=64
    max_dim=128
    num_layers=8
    sizes=[None]*(num_layers)
    conv_shapes=[(num_filters,max_dim)]*num_layers
    model=StiefelModel(sizes,None,convolutional=True,filter_length=5,dim=2,conv_shapes=conv_shapes,activation=act,scale_layer=False)
    pred=model(x_test[:100])
    model.fast_execution=True


    # adjust initialization
    for w in model.trainable_variables:
        w.assign(w/5./max_dim)

    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            noise=noise_level*tf.random.normal(tf.shape(batch))
            preds=model(batch+noise)
            val=tf.reduce_sum((preds-noise)**2)
        grad=tape.gradient(val,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        return val

    @tf.function
    def test_step(batch):
        noise=noise_level*tf.random.normal(tf.shape(batch))
        preds=model(batch+noise)
        err=tf.reduce_sum((preds-noise)**2)
        return err




    # Training
    train_ds=tf.data.Dataset.from_tensor_slices(x_train-y_train).shuffle(50000).batch(batch_size)
    test_ds=tf.data.Dataset.from_tensor_slices(x_test-y_test).shuffle(50000).batch(batch_size)
    epch=3000
    obj=0.
    backup_dir='results_conv/free_noise_level'+str(noise_level)
    if not os.path.isdir('results_conv'):
        os.mkdir('results_conv')
    if not os.path.isdir(backup_dir):
        os.mkdir(backup_dir)
    for batch in train_ds:
        obj+=test_step(batch)
    print(obj.numpy())   
    myfile=open(backup_dir+"/adam_log.txt","w")
    myfile.write("Log file for training\n")
    myfile.write(str(obj.numpy()))
    myfile.write("\n")
    myfile.close()
    for ep in range(epch):
        counter=0
        obj=0.
        for batch in train_ds:
            counter+=1
            val=train_step(batch)
            obj+=val
        print('Training Error: ' + str(obj.numpy()))
        myfile=open(backup_dir+"/adam_log.txt","a")
        myfile.write('Training Error: ' + str(obj.numpy())+'\n')
        myfile.close()
        obj=0.
        
        # save results
        with open(backup_dir+'/adam.pickle','wb') as f:
            pickle.dump(model.trainable_variables,f)
        if ep%5==0:
            with open(backup_dir+'/adam'+str(ep+1)+'.pickle','wb') as f:
                pickle.dump(model.trainable_variables,f)
            

