import pandas as pd
import numpy as np
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random

def parse_args():
    parser = ArgumentParser(description='anomly detection by netml')
    parser.add_argument('--input', type=str, help='path of input pcap or csv file')
    parser.add_argument('--raw', type=str, help='path of csv file')
    parser.add_argument('--output', type=str, help='path of output csv file')
    parser.add_argument('--iters', type=int, help='number of iterations')
    
    return parser.parse_args()

def df2timeseries(df, total_unit, start_time, end_time):
    
    timestamp = (df["time"] - start_time) / (end_time - start_time)
    weight = df["pkt_len"]
    weight = weight[(timestamp > 0) & (timestamp <= 1)]
    timestamp = timestamp[(timestamp > 0) & (timestamp <= 1)]
    timestamp = np.floor(timestamp * total_unit).astype(int)
    timestamp[timestamp == total_unit] = total_unit - 1
    
    packet_rate = np.bincount(timestamp)
    throughput = np.bincount(timestamp, weight)
    
    return packet_rate, throughput

def autoreg_train_predict(T_train, T_test, model, lags=5):
    
    X_train = np.array([T_train[i:i+lags] for i in range(len(T_train)-lags)]).copy()
    y_train = T_train[lags:].copy()
    
    model = make_pipeline(StandardScaler(), model)
    model.fit(X_train, y_train)
    
    X_test = T_train[-lags:].copy()
    y_test = T_test.copy()
    y_pred = np.zeros_like(y_test)
    
    for i in range(T_test.shape[0]):
        y_pred[i] = model.predict(X_test[np.newaxis, :])[0]
        
        X_test[:-1] = X_test[1:]
        X_test[-1] = y_pred[i]
    
    return y_pred

if __name__ == "__main__":

    args = parse_args()
    
    input_df = pd.read_csv(args.input)
    raw_df = pd.read_csv(args.raw)

    total_unit = 200
    se_unit = 100
    lags = 5
    pr, tput = df2timeseries(input_df, total_unit, start_time=raw_df["time"].min(), end_time=raw_df["time"].max())
    pr_raw, tput_raw = df2timeseries(raw_df, total_unit, start_time=raw_df["time"].min(), end_time=raw_df["time"].max())
    
    result_df = {
        "stat": [],
        "model": [],
        "MAE": []
    }
    
    stat_names = [
        "packet_rate",
        "throughput"
    ]
    stats_train = [
        pr,
        tput
    ]

    print(len(pr))
    stats_test = [
        pr_raw,
        tput_raw
    ]
    
    for i in range(5):
        random.seed(i)
        for stat_name, T_train_all, T_test_all in zip(stat_names, stats_train, stats_test):

            T_train = T_train_all[:total_unit//5*(i+1)][:-(total_unit//10)]
            T_test = T_test_all[:total_unit//5*(i+1)][-(total_unit//10):]

            model_names = [
                "AR",
                "ARMA"
            ]

            if T_train.size == 0:
                raise ValueError("T_train is empty. Check the data processing steps.")
    
            models = [
                ARIMA(T_train, order=(lags,0,0), enforce_stationarity=False),
                ARIMA(T_train, order=(lags,0,1), enforce_stationarity=False)
            ]
            
            for model_name, model in zip(model_names, models):
                result = model.fit()
                T_hat = result.forecast(steps=T_test.shape[0])
                
                MAE = (np.abs(T_hat - T_test)).sum()
                
                plt.plot(np.concatenate([T_train, T_test]))
                plt.plot(np.concatenate([T_train, T_hat]))
                os.makedirs(args.output[:-4], exist_ok=True)
                plt.savefig(os.path.join(args.output[:-4],stat_name+model_name+str(i)+".svg"), format="svg")
                plt.clf()
                
                result_df["stat"].append(stat_name)
                result_df["model"].append(model_name)
                result_df["MAE"].append(MAE)
                
                print(stat_name + model_name + ": " + str(MAE))
            
            model_names = [
                "KNN",
                "DT",
                "RF",
                "AB"
            ]
            models = [
                KNeighborsRegressor(5),
                DecisionTreeRegressor(max_depth=5, random_state=6),
                RandomForestRegressor(
                    max_depth=5, n_estimators=10, max_features=1, random_state=6
                ),
                AdaBoostRegressor(random_state=6)
            ]
            
            for model_name, model in zip(model_names, models):
                T_hat = autoreg_train_predict(T_train, T_test, model, lags=lags)
                
                MAE = (np.abs(T_hat - T_test)).sum()
                
                plt.plot(np.concatenate([T_train, T_test]))
                plt.plot(np.concatenate([T_train, T_hat]))
                os.makedirs(args.output[:-4], exist_ok=True)
                plt.savefig(os.path.join(args.output[:-4],stat_name+model_name+str(i)+".svg"), format="svg")
                plt.clf()
                
                result_df["stat"].append(stat_name)
                result_df["model"].append(model_name)
                result_df["MAE"].append(MAE)
                
                print(stat_name + model_name + ": " + str(MAE))
    
    result_df = pd.DataFrame(result_df)
    result_df.to_csv(args.output, index=False)
    
        
        