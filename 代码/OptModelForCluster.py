'''
Clusters	Selected optimal model	Fitness
1	          RF	                 0.8924
2	          GPR	                 0.9595
3	          GPR	                 0.9376
4	          RF	                 0.9692
5	          RF	                 0.9364
6	          SVR	                 0.9548
7	          SVR	                 0.9791
8	          RF	                 0.9212
'''
import pandas as pd


def gaussian_model():
    #GPR model
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
    return model, parameter


def svr_model():
    #SVR model
    from sklearn.svm import SVR
    parameter = ' svr_model '
    model = SVR(kernel='rbf', C=100,  gamma='auto')
    return model, parameter


def random_forest_model():
    #random forest model
    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForest_model '
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)\
    #10 ,6
    model = RandomForestRegressor(n_estimators=15, max_depth=6, criterion='mae', bootstrap=True)
    return model, parameter

def linear_model():
    from sklearn.linear_model import LinearRegression
    parameter = ' Linear_model '
    model = LinearRegression(normalize=True)

    return model, parameter


def lasso_model():
    from sklearn.linear_model import Ridge, Lasso
    parameter = 'Lasso_model'
    model = Lasso(alpha=0.01)
    return model, parameter


def ridge_model():
    from sklearn.linear_model import Ridge
    parameter = ' Elastic_model '
    model = Ridge(alpha = 0.02)
    return model, parameter


import pickle
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
all_clusters = ["cluster_0", "cluster_1", "cluster_2",
               "cluster_3", "cluster_4",
               "cluster_5","cluster_6","cluster_7"] #eight different
# alloy clusters with various creep mechanisms
candidate_models = ["RF", "GPR", "SVR", "LR", "RR"]#three candidate machine learning model
cluster_model = {"cluster_0": "SVR", "cluster_1": "RR","cluster_2": "RR",
                 "cluster_3": "GPR","cluster_4": "RF","cluster_5": "RF",
                 "cluster_6": "SVR","cluster_7": "SVR"}
print("all models start running!")
for cluster in cluster_model:
    print(cluster)
    df = pd.read_excel("data_files/multi-factors_clusters_912.xls",sheetname=cluster)
    cluster_sample = df.values[:, 1:]
    # print("11111", cluster_sample.shape)
    data, target = cluster_sample[:, 0:22], cluster_sample[:, -1]
    # print("2222", data)
    df_original = pd.read_excel("data_files/多尺度样本_revision.xlsx")
    original_data = df_original.values[:, 1:23]
    # original_data = df_original.values[:, 1:-1]
    # print(original_data.shape)
    # original_min = np.min(original_data, axis=0)
    # original_max = np.max(original_data, axis=0)
    # train_dataX = [(row-original_min)/(original_max-original_min)
    #               for row in original_dataX]
    # train_dataX = np.array(train_dataX)
    scaler = MinMaxScaler()
    scaler.fit(original_data)  # 用全部266条数据的最大与最小值对每个簇上的特征进行最大最小归一化
    data = scaler.transform(data)
    # print("归一化后的值", data)
    target = np.log(target)
    # print(data ,target)
    if cluster_model.get(cluster) == "RF":
        rf_model, rf_para = random_forest_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers1/"+cluster+"RF"+".model")
    elif cluster_model.get(cluster) == "GPR":
        rf_model, rf_para = gaussian_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers1/"+cluster + "GPR" + ".model")
    elif cluster_model.get(cluster) == "SVR":
        rf_model, rf_para = svr_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers1/"+cluster + "SVR" + ".model")
    elif cluster_model.get(cluster) == "LR":
        lasso_model, lr_para = lasso_model()
        lasso_model.fit(data, target)
        joblib.dump(lasso_model, "model_for_custers1/" + cluster + "LR" + ".model")
    else:
        rr_model, rr_para = ridge_model()
        rr_model.fit(data, target)
        joblib.dump(rr_model, "model_for_custers1/" + cluster + "RR" + ".model")
print("all models have already save!")