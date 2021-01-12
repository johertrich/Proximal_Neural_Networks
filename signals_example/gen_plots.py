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

def run():
    x_train,y_train,x_test,y_test=loadData('signals',500000)

    x_train=1.0*np.array(x_train).astype(np.float32)
    y_train=1.0*np.array(y_train).astype(np.float32)
    x_test=1.0*np.array(x_test).astype(np.float32)
    y_test=1.0*np.array(y_test).astype(np.float32)

    if not os.path.exists('results'):
        raise NameError('Directory results is missing!')
    myfile=open('results_signals/psnrs_imgs.txt',"w")
    myfile.write("mean PSNR of noisy images: "+str(meanPSNR(x_test,y_test)))
    myfile.write("\nMSE of noisy images: "+str(MSE(x_test,y_test)))
    myfile.write("\nPSNR noisy 1: "+str(meanPSNR(x_test[4,:],y_test[4,:])))
    myfile.write("\nPSNR noisy 2: "+str(meanPSNR(x_test[2,:],y_test[2,:])))
    plotSaveSignal(y_test[4,:],'results_signals/orig1.png',(-1.8,1.8))
    plotSaveSignal(y_test[2,:],'results_signals/orig2.png',(-0.7,0.5))
    plotSaveSignal(x_test[4,:],'results_signals/noisy1.png',(-1.8,1.8))
    plotSaveSignal(x_test[2,:],'results_signals/noisy2.png',(-0.7,0.5))
    model_names=['small_1layer','small_2layer','small_3layer','big_1layer','big_2layer','big_3layer','haar_small_0.1','haar_big_0.03','haar_scaled_0.08','haar_small_0.3','haar_small_learned','haar_big_learned','haar_scaled_learned']
    out_names=['small1L','small2L','small3L','big1L','big2L','big3L','haar_small_thresh0_1','haar_big_thresh0_03','haar_scaled_thresh0_08','haar_small_thresh0_3','haar_small_learned','haar_big_learned','haar_big_scaled_learned']
    my_len=len(model_names)
    for i in range(my_len):
        print(str(i)+' of '+str(my_len))
        name=model_names[i]
        out_name=out_names[i]
        model=tf.keras.models.load_model('results_signals/'+name)
        pred=model(x_test)
        myfile.write('\n\nModel '+name+':')
        if hasattr(model,'stiefel'):
            if hasattr(model.stiefel[0],'soft_thresh'):
                myfile.write('\nlearned thresholds:')
                for i in range(len(model.stiefel)):
                    if model.stiefel[i].soft_thresh.numpy()<0:
                        myfile.write('\nLayer '+str(i)+': '+str(np.exp(model.stiefel[i].soft_thresh.numpy())))
                    else:
                        myfile.write('\nLayer '+str(i)+': '+str(model.stiefel[i].soft_thresh.numpy()**2))
        myfile.write('\nPSNR 1: '+str(meanPSNR(pred[4,:],y_test[4,:])))
        myfile.write('\nPSNR 2: '+str(meanPSNR(pred[2,:],y_test[2,:])))
        myfile.write('\nmean PSNR: '+str(meanPSNR(pred,y_test)))
        myfile.write("\nMSE: "+str(MSE(pred,y_test)))
        plotSaveSignal(pred[4,:],'results_signals/'+out_name+'1.png',(-1.8,1.8))
        plotSaveSignal(pred[2,:],'results_signals/'+out_name+'2.png',(-0.7,0.5))
    myfile.close()
