# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
# In the following code, a neural network with the same structure as the 
# PNN in MNIST_PNN.py is trained.
#
import numpy as np
import numpy.matlib
import tensorflow as tf
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

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

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # declare model
    class MyModel(Model):
      def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(784, activation='relu')
        self.d2 = Dense(784, activation='relu')
        self.d3 = Dense(400, activation='relu')
        self.d4 = Dense(400, activation='relu')
        self.d5 = Dense(200, activation='relu')
        self.d6 = Dense(10)

      def call(self, x):
        x = self.d6(self.d5(self.d4(self.d3(self.d2(self.d1(x))))))
        return 1/(1+tf.exp(-x))

    # Create an instance of the model
    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # train model
    @tf.function
    def train_step(images, labels):
      with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        for weight in model.weights:
            loss+=1e-6*tf.reduce_sum(weight*weight)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)
      train_accuracy(tf.math.argmax(labels,axis=1), predictions)

    @tf.function
    def test_step(images, labels):
      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)

      test_loss(t_loss)
      test_accuracy(tf.math.argmax(labels,axis=1), predictions)

    EPOCHS = 1000
    maxim_acc=0.
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()
      step=0    

      for images, labels in train_ds:
        train_step(images, labels)

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        
      maxim_acc=tf.reduce_max([maxim_acc,test_accuracy.result()])
      template = 'Epoch {0}, Loss: {1:1.6f}, Accuracy: {2:2.2f}, Test Loss: {3:2.4f}, Test Accuracy: {4:2.2f}, Maximal Test Accuracy: {5:2.2f}'
      print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100,
                            maxim_acc*100))
    # save model
    model.save('results_MNIST/MNIST_comparison')
