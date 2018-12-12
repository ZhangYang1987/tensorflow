#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:02:23 2018

@author: apple
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import plot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
#data=pd.read_table('/Data/zhengqi_train.txt')    


#selector = VarianceThreshold(0.1)
#x_vt=selector.fit_transform(x)
#
#def main():
#    x=data.iloc[:,0:-1]
#    y=data.iloc[:,-1]
#    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#    rf=RandomForestRegressor()
#    rf.fit(x_train,y_train)
#    r2_train=rf.score(x_train,y_train)
#    r2_test=r2_score(y_test,rf.predict(x_test))
    
def BP():
    data=pd.read_table('/Data/zhengqi_train.txt')    
    x=data.iloc[:,0:-1]
    y=data.iloc[:,-1]

    input_size=38
    hidden_size=10
    output_size=1
    batch_size=100
    
    
    X=tf.placeholder(dtype=tf.float32,shape=[None,input_size])
    Y=tf.placeholder(dtype=tf.float32,shape=[None,output_size])
    
    w1=tf.Variable(np.random.rand(input_size,hidden_size).astype(np.float32))
    b1=tf.Variable(np.random.rand(hidden_size).astype(np.float32))
    w2=tf.Variable(np.random.rand(hidden_size,output_size).astype(np.float32))
    b2=tf.Variable(np.random.rand(output_size).astype(np.float32))
    hidden_out=tf.matmul(X,w1)+b1
    h=tf.sigmoid(hidden_out)
    output=tf.matmul(h,w2)+b2
    
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        random_choice=np.random.choice(500,10)
        sess.run(output,feed_dict={X:x.iloc[random_choice,:]})
#        sess.run(output,feed_dict={X:x})
        y_pre=list(output)
        
    return y_pre
    
    
    

    