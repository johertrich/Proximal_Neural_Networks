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
import os.path

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

def run():
    if not os.path.exists('results_signals'):
        os.mkdir('results_signals')

    # load and preprocess data
    x_train,y_train,x_test,y_test=loadData('signals',500000)

    x_train=1.0*np.array(x_train).astype(np.float32)
    y_train=1.0*np.array(y_train).astype(np.float32)
    x_test=1.0*np.array(x_test).astype(np.float32)
    y_test=1.0*np.array(y_test).astype(np.float32)

    # set parameters and create log file
    epochs=125

    myfile=open("results_signals/psnrs_orth.txt","w")
    myfile.write("mean PSNR of noisy images: "+str(meanPSNR(x_test,y_test)))
    myfile.write("\nPSNR noisy 1: "+str(meanPSNR(x_test[1,:],y_test[1,:])))
    myfile.write("\nPSNR noisy 2: "+str(meanPSNR(x_test[2,:],y_test[2,:])))
    myfile.close()


    thresh=None
    # define model
    model=StiefelModel([128],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/small1L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/small1L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_small_1L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nOne layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    model.save('results_signals/small_1layer')


    # define model
    model=StiefelModel([128,128],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/small2L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/small2L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_small_2L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nTwo layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    myfile.close()
    model.save('results_signals/small_2layer')

    # define model
    model=StiefelModel([128,128,128],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/small3L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/small3L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_small_2L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nTwo layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    myfile.close()
    model.save('results_signals/small_3layer')

    # define model
    model=StiefelModel([1024],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/big1L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/big1L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_big_1L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nOne big layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    myfile.close()
    model.save('results_signals/big_1layer')

    # define model
    model=StiefelModel([1024,1024],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/big2L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/big2L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_big_2L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nTwo big layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    myfile.close()
    model.save('results_signals/big_2layer')

    # define model
    model=StiefelModel([1024,1024,1024],thresh)
    # train model
    train(model,x_train,y_train,x_test,y_test,epochs,learn_rate=0.5)
    # test model and create outputs
    pred=model(x_test)
    plotSaveSignal(pred[1,:],'results_signals/big3L_thresh'+str(thresh)+'1.png')
    plotSaveSignal(pred[2,:],'results_signals/big3L_thresh'+str(thresh)+'2.png')
    os.rename('log.txt','results_signals/training_log_big_3L_thresh'+str(thresh)+'.txt')
    myfile=open("results_signals/psnrs_orth.txt","a")
    myfile.write("\n\nTwo big layer threshold "+str(thresh)+"\n")
    myfile.write("\nPSNR 1: "+str(meanPSNR(pred[1,:],y_test[1,:])))
    myfile.write("\nPSNR 2: "+str(meanPSNR(pred[2,:],y_test[2,:])))
    myfile.write("\nPSNR all: "+str(meanPSNR(pred,y_test)))
    myfile.close()
    model.save('results_signals/big_3layer')
