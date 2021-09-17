# This code belongs to the paper
# 
# J. Hertrich, S. Neumayer and G. Steidl.  
# Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.
# Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.
#
# Please cite the paper if you use this code.
#
# This file contains implementations for the FBS-PnP and the ADMM-PnP.
#
import tensorflow as tf
from PIL import Image
import time

def f_standard(signal,inp_signal):
    return 0.5 *tf.reduce_sum((signal-inp_signal)**2)

def T_standard(inp,model):
    return model(inp)

def T_residual(inp,model):
    return inp-model(inp)

def PnP_FBS(model,inputs,fun=f_standard,tau=0.9,eps=1e-5,T_fun=T_standard,grad_f=None,grad_steps=1,truth=None):
    inp_signal=tf.constant(inputs,dtype=tf.float32)
    if grad_f is None:
        f=lambda signal: fun(signal,inp_signal)
        def grad_f(signal):
            with tf.GradientTape() as tape:
                tape.watch(signal)
                val=f(signal)
            grad=tape.gradient(val,signal)
            return grad
    T=lambda inp: T_fun(inp,model)
    max_steps=800
    my_signal=inp_signal
    prev_step=tf.identity(my_signal)
    for it in range(max_steps):
        for st in range(grad_steps):
            grad=grad_f(my_signal)
            my_signal-=tau*grad
        my_signal2=tf.identity(my_signal)
        my_signal=T(my_signal)
        norm_change=tf.reduce_sum((prev_step-my_signal)**2).numpy()
        if norm_change<eps:
            break
        prev_step=tf.identity(my_signal)
    return my_signal


def prox_f_standard(signal,lam,inp_signal):
    return 1.0/(lam+1.0)*(lam*signal+inp_signal)

def PnP_ADMM(inputs,T_fun,gamma=11,prox_fun=prox_f_standard,eps=1e-3):
    inp_signal=tf.constant(inputs,dtype=tf.float32)
    prox_f=lambda signal,lam:prox_fun(signal,lam,inp_signal)
    T=T_fun
    max_steps=300
    x=tf.identity(inp_signal)
    x_old=0
    y=tf.identity(inp_signal)
    p=0
    gamma=tf.constant(gamma,dtype=tf.float32)
        
    for it in range(max_steps):
        y=T(1./gamma*p+x)
        p+=gamma*(x-y)
        x=prox_f(y-1./gamma*p,gamma)
        norm_change=tf.reduce_sum((x_old-x)**2).numpy()
        
        if norm_change<eps:
            break
        x_old=tf.identity(x)
    return x
