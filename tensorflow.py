#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:32:13 2018

@author: apple
"""

import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor as sp_RF
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
spark = SparkSession.builder\
    .master("local")\
    .appName("Test")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()
#data=pd.read_table('/Data/zhengqi_train.txt')    
def sp(data):
    df=spark.createDataFrame(data)
    x_columns=df.columns[0:-1]
    y_columns=df.columns[-1]
    VA=VectorAssembler(inputCols=x_columns,outputCol='features')
    feature_assemble=VA.transform(df)
    rf=sp_RF()
    grid = ParamGridBuilder().baseOn({rf.labelCol:'target'}).addGrid(rf.numTrees, [10, 100]).build()
    evaluator = RegressionEvaluator(labelCol='target')
    cv = CrossValidator(estimator=rf,estimatorParamMaps=grid,evaluator=evaluator)
    cvModel = cv.fit(feature_assemble)
    return cvModel

    
    
#data=pd.read_excel('/Data/副本减二减三收率预测.xlsx')
#data=pd.read_excel('/Data/副本所有收率.xlsx')
#data=pd.read_excel('/Data/yuanyou1205test.xlsx')
def preprocess(data):
    
    df=data.fillna(0)
    return df

def pipeline(df):
    x=df[df.columns[1:-2]]
    y=df[df.columns[-2]]
    split=int(len(df)*0.9)
    x_train=x.iloc[0:split]
    y_train=y.iloc[0:split]
    x_pre=x.iloc[split:]
    y_pre=y.iloc[split:]
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.1)
    rf=RandomForestRegressor()
    scores=cross_validate(rf,x_train,y_train,scoring='r2',cv=5)
    parameters={'n_estimators':(10,20,30,40)}
    clf=GridSearchCV(rf,parameters)
    clf.fit(x_train,y_train)
    best_estimator=clf.best_estimator_
#    rf_PL=Pipeline(steps=[('rf',rf)])
    rf_PL=Pipeline(steps=[('rf',best_estimator)])
#    rf_PL.set_params(rf__n_estimators=10).fit(x_train,y_train)
#    r2=rf_PL.score(x_train,y_train)
    pre=rf_PL.predict(x_pre)
    r2_test=r2_score(y_pre,pre)    
#    return x_train,x_test,y_train,y_test
    result={}
    result['best_estimator']=best_estimator
    result['pre']=pre
    result['real']=y_test
    result['scores']=scores
    result['validation_r2']=r2_test
    return result