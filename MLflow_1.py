import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
df=pd.read_table('/Data/zhengqi_train.txt') 

def main(df):
    x_train=df.iloc[:,0:-1]
    y_train=df.iloc[:,-1]
    rf=RandomForestRegressor()
    rf.fit(x_train,y_train)
    score=rf.score(x_train,y_train)
    result=rf.estimators_
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.log_param("n_estimators",10)
    mlflow.sklearn.log_model(rf, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    return result


if __name__=="__main__":
    main(df)
