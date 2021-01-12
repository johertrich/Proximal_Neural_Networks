# This code belongs to the paper
#
# M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.
# Parseval proximal neural networks.  
# Journal of Fourier Analysis and Applications, 26:59, 2020.
#
# Please cite the paper if you use this code.
#
from core.stiefel_network import *
import os

class ScaledHaarModel(Model):
    # Inherits from tensorflow.keras.Model.
    # Executes a denoising of signals using scaled Haar wavelets
    def __init__(self,my_lambda):
        super(ScaledHaarModel,self).__init__()
        if my_lambda is None:
            self.train_lambda=True
            self.my_lambda=self.add_weight("lambda",initializer=tf.constant_initializer(np.log(0.1)),shape=[1],trainable=True)
        else:        
            self.my_lambda=tf.constant(my_lambda)
            self.train_lambda=False
        self.haMat=tf.constant(haarWavelets(7),dtype=tf.float32)
        factor_vector=np.array([])
        for i in range(0,8):
            factor_vector=np.hstack([factor_vector, 1/np.power(np.sqrt(2),i)*np.ones(128)])
        self.factor_vector=tf.constant(factor_vector,dtype=tf.float32)
    
    def call(self,inputs):
        if inputs is None:
            return None         
        haMat=haarWavelets(7)
        soft_thresh=self.my_lambda
        if self.train_lambda:
            soft_thresh=tf.exp(self.my_lambda)
        fac_vec=1/self.factor_vector
        fac_vec=tf.expand_dims(fac_vec,-1)
        my_dim=inputs.shape[0]
        if my_dim is None:
            my_dim=1000
        fac_vec=tf.tile(fac_vec,(1,my_dim))
        pred_haar_3=tf.matmul(self.haMat,tf.transpose(inputs))
        pred_haar_3=fac_vec*pred_haar_3
        pred_haar_3=tf.math.sign(pred_haar_3)*tf.maximum(tf.math.abs(pred_haar_3)-soft_thresh,0)
        pred_haar_3=pred_haar_3/fac_vec
        pred_haar_3=tf.transpose(tf.matmul(tf.transpose(self.haMat),pred_haar_3))
        return pred_haar_3

def haarWavelet(power):
    width=np.power(2,power)
    out=np.ones(width)/np.sqrt(width)
    for i in range(1,power+1):
        leng=np.power(2,power-i)
        fil=np.hstack([np.ones(leng),-np.ones(leng)])
        links=0
        while links<np.power(2,power):
            line=np.hstack([np.zeros(links), fil, np.zeros(width-links-2*leng)])/np.sqrt(2*leng)
            out=np.vstack([out,line])
            links=links+2*leng
    return out

def haarWavelets(power):
    width=np.power(2,power)
    out=[]
    for i in range(0,power):
        line=np.hstack([np.ones(np.power(2,i)),-np.ones(np.power(2,i)),np.zeros(width-np.power(2,i+1))])
        line=line/np.power(2,i)
        for j in range(0,width):
            out.append(line)
            line=np.hstack([line[-1], line[0:-1]])
    for i in range(0,width):
        out.append(np.ones(width)/np.power(2,power-1))
    out=np.vstack(out)/2
    return out

def cv_haar(x_train,y_train,lambda_candidates,parts,haMat):
    n=x_train.shape[0];
    mses=[]
    for lam_ind in range(0,len(lambda_candidates)):
        lam=lambda_candidates[lam_ind];
        perm=np.random.permutation(n)
        sum_mse=0
        for i in range(0,parts):
            model=StiefelModel([haMat.shape[0]],lam)
            break1=round(i*n*1.0/parts)
            break2=round((i+1)*n*1.0/parts)
            tr_inds=np.hstack([perm[0:break1],perm[break2:]])
            te_inds=perm[break1:break2]
            x_tr=x_train[tr_inds,:]
            y_tr=y_train[tr_inds,:]
            y_te=y_train[te_inds,:]
            x_te=x_train[te_inds,:]
            model(x_tr[:10,:])
            model.stiefel[0].matrix.assign(haMat)
            pred=model(x_te)
            my_mse=np.sum(np.sum((pred-y_te)*(pred-y_te)))/y_te.shape[0]/y_te.shape[1]
            sum_mse+=my_mse
        mses.append(sum_mse/parts);
    best_ind=np.argmin(np.array(mses))
    return lambda_candidates[best_ind]

def cv_haar2(x_train,y_train,lambda_candidates,parts,haMat):
    n=x_train.shape[0];
    mses=[]
    for lam_ind in range(0,len(lambda_candidates)):
        lam=lambda_candidates[lam_ind];
        perm=np.random.permutation(n)
        sum_mse=0
        for i in range(0,parts):
            break1=round(i*n*1.0/parts)
            break2=round((i+1)*n*1.0/parts)
            tr_inds=np.hstack([perm[0:break1],perm[break2:]])
            te_inds=perm[break1:break2]
            x_tr=x_train[tr_inds,:]
            y_tr=y_train[tr_inds,:]
            y_te=y_train[te_inds,:]
            x_te=x_train[te_inds,:]
            soft_thresh=lam
            factor_vector=np.array([])
            for i in range(0,8):
                factor_vector=np.hstack([factor_vector, 1/np.power(np.sqrt(2),i)*np.ones(128)])
            pred_haar2=[]
            for i in range(0,len(x_te)):
                pred_haar_2=1/factor_vector*haMat.dot(x_te[i])
                pred_haar_2=np.sign(pred_haar_2)*np.maximum(np.abs(pred_haar_2)-soft_thresh,0) 
                pred_haar_2=pred_haar_2*factor_vector
                pred_haar2.append(haMat.transpose().dot(pred_haar_2))
            pred_haar2=np.array(pred_haar2)
            pred=pred_haar2
            my_mse=np.sum(np.sum((pred-y_te)*(pred-y_te)))/y_te.shape[0]/y_te.shape[1]
            sum_mse+=my_mse
        mses.append(sum_mse/parts);
    best_ind=np.argmin(np.array(mses))
    return lambda_candidates[best_ind]

def run():
    # load and preprocess data
    x_train,y_train,x_test,y_test=loadData('signals',500000)

    x_train=1.0*np.array(x_train).astype(np.float32)
    y_train=1.0*np.array(y_train).astype(np.float32)
    x_test=1.0*np.array(x_test).astype(np.float32)
    y_test=1.0*np.array(y_test).astype(np.float32)


    # 128 x 128 Matrices:
    thresholds=[0.05,0.1,0.15,0.2,0.25,0.3]
    haMat=haarWavelet(7)
    # determine optimal threshhold lambda_cv via cross validation
    lambda_cv=cv_haar(x_train,y_train,thresholds,5,haMat);
    # create model with optimal threshhold lambda_cv
    model=StiefelModel([haMat.shape[0]],lambda_cv)
    model(x_train[:10,:])
    model.stiefel[0].matrix.assign(haMat)
    # test model and save plots
    pred=model(x_test)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    plotSaveSignal(pred[4,:],'results_signals/haar_basis_thresh'+str(lambda_cv)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/haar_basis_thresh'+str(lambda_cv)+'2.png')
    print("Haar Basis: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(lambda_cv))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model._set_inputs(x_test)
    model.save('results_signals/haar_orth_'+str(lambda_cv))

    # Test model and save plots with threshhold 0.3
    lambda_cv=0.3
    haMat=haarWavelet(7)
    model=StiefelModel([haMat.shape[0]],lambda_cv)
    model(x_train[:10,:])
    model.stiefel[0].matrix.assign(haMat)
    model._set_inputs(x_test)
    model.save('results_signals/haar_orth_'+str(lambda_cv))

    # 1024 x 128 matrices using Haar wavelets
    thresholds=[0.01,0.02,0.03,0.04,0.05,0.06]
    haMat=haarWavelets(7)
    # determine optimal threshhold lambda_cv via cross validation
    lambda_cv=cv_haar(x_train,y_train,thresholds,5,haMat);    
    # create model with optimal threshhold lambda_cv
    model=StiefelModel([haMat.shape[0]],lambda_cv)
    model(x_train[:10,:])
    model.stiefel[0].matrix.assign(haMat)
    # test model and save plots
    pred=model(x_test)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    plotSaveSignal(pred[4,:],'results_signals/haar_frame_thresh'+str(lambda_cv)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/haar_frame_thresh'+str(lambda_cv)+'2.png')
    print("Haar Wavelet: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(lambda_cv))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model._set_inputs(x_test)
    model.save('results_signals/haar_big_'+str(lambda_cv))


    # 1024 x 128 matrices using scaled Haar wavelets
    thresholds=[0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    haMat=haarWavelets(7)
    # determine optimal threshhold lambda_cv via cross validation
    lambda_cv=cv_haar2(x_train,y_train,thresholds,5,haMat);
    soft_thresh=lambda_cv
    factor_vector=np.array([])
    for i in range(0,8):
        factor_vector=np.hstack([factor_vector, 1/np.power(np.sqrt(2),i)*np.ones(128)])
    pred_haar2=[]
    for i in range(0,len(x_test)):
        pred_haar_2=1/factor_vector*haMat.dot(x_test[i])
        pred_haar_2=np.sign(pred_haar_2)*np.maximum(np.abs(pred_haar_2)-soft_thresh,0) 
        pred_haar_2=pred_haar_2*factor_vector
        pred_haar2.append(haMat.transpose().dot(pred_haar_2))
    pred_haar2=np.array(pred_haar2)
    pred=pred_haar2   
    # create model with optimal threshhold lambda_cv 
    model=ScaledHaarModel(my_lambda=lambda_cv)
    # test model and save plots
    pred2=model(x_test)
    print(pred2.shape)
    mse=np.sum(np.sum((pred-y_test)*(pred-y_test)))/y_test.shape[0]/y_test.shape[1]
    psnr=meanPSNR(pred,y_test)
    plotSaveSignal(pred[4,:],'results_signals/haar_frame_scaled_thresh'+str(lambda_cv)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/haar_frame_scaled_thresh'+str(lambda_cv)+'2.png')
    print("Haar Wavelet scaled: PSNR "+str(psnr)+" MSE: "+str(mse))
    print("Best Threshold: "+str(lambda_cv))
    print("\nPSNR 1: "+str(meanPSNR(pred[4,:],y_test[4,:])))
    print("\nMSE 1: "+str(MSE(pred[4,:],y_test[4,:])))
    print("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    print("\nMSE 2: "+str(MSE(pred[2,:],y_test[2,:])))
    model._set_inputs(x_test)
    model.save('results_signals/haar_scaled_'+str(lambda_cv))



