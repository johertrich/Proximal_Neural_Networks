# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
from core.stiefel_network import *
from signals_example.haar_cv import *
import os

def run():
    x_train,y_train,x_test,y_test=loadData('signals',500000)

    x_train=1.0*np.array(x_train).astype(np.float32)
    y_train=1.0*np.array(y_train).astype(np.float32)
    x_test=1.0*np.array(x_test).astype(np.float32)
    y_test=1.0*np.array(y_test).astype(np.float32)

    def train_thresh(model,x_train,y_train,x_test,y_test,EPOCHS=5,learn_rate=0.5,batch_size=32,model_type='normal'):
        # function to learn the optimal threshhold in the Haar wavelet denoising.
        loss_object=tf.keras.losses.MeanSquaredError()
        train_loss=tf.keras.metrics.Mean(name="train_loss")
        test_loss=tf.keras.metrics.Mean(name="test_loss")
        test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)
        train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(x_train.shape[0]).batch(batch_size)

        @tf.function
        def train_step(inputs,outputs):
            with tf.GradientTape() as tape:
                predictions=model(inputs)
                loss=loss_object(outputs,predictions)
            if model_type=='scaled':
                gradient=tape.gradient(loss,model.my_lambda)
                model.my_lambda.assign_sub(learn_rate*gradient)
            else:
                gradient=tape.gradient(loss, model.stiefel[0].soft_thresh)
                model.stiefel[0].soft_thresh.assign_sub(learn_rate*gradient)
            train_loss(loss)
        @tf.function
        def test_step(inputs,outputs):
            predictions=model(inputs)
            t_loss=loss_object(outputs,predictions)
            test_loss(t_loss)
        
        for epoch in range(EPOCHS):
            print('Epoch '+str(epoch))
            for inputs,outputs in train_ds:
                train_step(inputs,outputs)
            for test_inputs,test_outputs in test_ds:
                test_step(test_inputs,test_outputs)
            print('Loss: '+str(float(train_loss.result()))+', Test Loss: '+str(float(test_loss.result())),end=', ')     
            pred=model(x_test)
            err=np.sum(((pred-y_test)*(pred-y_test)).numpy())/len(x_test)
            psnr_test=meanPSNR(pred,y_test)
            print('MSE: ' +str(err)+' PSNR: '+str(psnr_test))

    # create model
    haMat=haarWavelet(7)
    model=StiefelModel([haMat.shape[0]],None)
    model(x_train[:10,:])
    model.stiefel[0].matrix.assign(haMat)
    # train threshhold
    train_thresh(model,x_train,y_train,x_test,y_test,EPOCHS=5)
    # test and create plots
    pred=model(x_test)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    print("Haar Basis: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(np.exp(model.stiefel[0].soft_thresh.numpy())))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model.save('results_signals/haar_small_learned')

    # create model
    haMat=haarWavelets(7)
    model=StiefelModel([haMat.shape[0]],None)
    model(x_train[:10,:])
    model.stiefel[0].matrix.assign(haMat)
    # train threshhold
    train_thresh(model,x_train,y_train,x_test,y_test,EPOCHS=5)
    # test and create plots
    pred=model(x_test)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    print("Haar Wavelet: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(np.exp(model.stiefel[0].soft_thresh.numpy())))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model.save('results_signals/haar_big_learned')


    # create model
    haMat=haarWavelets(7)
    model=ScaledHaarModel(my_lambda=None)
    pred=model(x_test[:10,:])
    # train threshhold
    train_thresh(model,x_train,y_train,x_test,y_test,EPOCHS=5,model_type='scaled')
    # test and create plots
    pred=model(x_test)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    print("Haar Wavelet scaled: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(np.exp(model.my_lambda.numpy())))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model.save('results_signals/haar_scaled_learned')


