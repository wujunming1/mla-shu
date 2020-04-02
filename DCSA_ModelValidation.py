'''
65.89	5.61	0.99	5.2	0	7.56	0.81	4.27	8.52	0.07	0.03	0	0.83	0.22	4	4	20	1280	1080	871	1040	137.2	61.06426846	4.51733E-21	53.96456487	0.350781578	0.75214	454.3
58.832	1.26	12.4	4.35	0	6.32	0.73	6.6	7.68	0.062	0.026	0	1.5	0.24	4	4	20	1270	1080	871	1040	137.2	59.77759412	2.24667E-21	57.57985457	0.357581635	0.54641	211.1
64.53	0	5	4.15	3.68	6.5	0.92	10.6	4.42	0	0	0	0	0.2	3	4	20	1276	1080	871	204	759	41.04733707	1.80716E-21	69.04909109	0.349344299	0.60239	256.2
64.73	0	5	4.15	3.68	6.5	0.92	10.6	4.42	0	0	0	0	0	3	4	20	1288	1080	871	315	448.5	41.34221675	1.79973E-21	69.20082315	0.349701573	0.5966	277.1
'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
# data = pd.read_excel("data_files/多尺度样本聚类_912.xls", sheetname="cluster_6").values
# X = data[:, :-1]
# y = np.log(data[:, -1])
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
# model = RandomForestRegressor(n_estimators=15, max_depth=6, criterion='mae', bootstrap=True)
# model.fit(X, y)
# pred_data = pd.read_excel("test_data_file/8-new validated instance.xlsx").values
# single_data = pred_data[6:7,:-1]
# print(single_data)
# single_data = scaler.transform(single_data)
# print("22222", single_data)
# pred_y = model.predict(single_data)
# print("1111tian", np.exp(pred_y))
# a=np.array([[1,2,3],
#             [2,3,5],
#             [7,8,9]])
# print(MinMaxScaler().fit_transform(a))
# max_a = np.max(a, axis=0)
# min_a = np.min(a, axis=0)
# # a[0](max_a-min_a)
# norm_a = [(row-min_a)/(max_a-min_a) for row in a]
# print(norm_a)
def pre_progressing(original_data):
    data = original_data[:, :-1]
    target = original_data[:, -1]
    normalize_data = MinMaxScaler().fit_transform(data)
    normalize_target = np.log(target)
    return normalize_data, normalize_target

df_original = pd.read_excel("data_files/多尺度样本_revision.xlsx")
original_data = df_original.values[:, 1:-1]
print(original_data)
original_min = np.min(original_data, axis=0)
original_max = np.max(original_data, axis=0)
print("11111", original_max)
print("22222", original_min)
print(original_min.shape)
print(original_max-original_min)

cluster_centers = []
for index in range(8):
    #每个簇上的样本
    cluster_data = pd.read_excel("data_files/multi-factors_clusters_912.xls",
                                 sheetname="cluster_"+str(index))
    print("222ddddd", cluster_data.values)
    cluster_dataX = cluster_data.values[:, 1:28]
    normalize_data = [(row-original_min)/(original_max-original_min)
                      for row in cluster_dataX]#每个簇按照全部最大最小值归一化
    #求每个簇的聚类中心
    print("每个簇最大最小归一化的数据为", normalize_data)
    cluster_center = np.mean(normalize_data, axis=0)
    cluster_centers.append(cluster_center)
print("聚类中心为", cluster_centers)
cluster_centers = np.array(cluster_centers)#将聚类中心转化为numpy数组
#8条验证样本
df = pd.read_excel("test_data_file/8-new validated instance_revison.xlsx")
validation_data = df.values
# print(validation_data)
validation_dataX, validation_targetY = validation_data[:, :-1], np.log(validation_data[:, -1])
# normalize_data, normalize_target = pre_progressing(validation_data)
#第一条样本
data, target = (validation_dataX[7:8]-original_min)/(original_max-original_min), validation_targetY[7:8]
cluster_distance = [np.sqrt(np.sum((data-value)**2, axis=1))
                    for value in cluster_centers]#样本到各个簇之间的距离
print("样本到各个簇上的距离", cluster_distance)
cluster_dis = [dis[0] for dis in cluster_distance]
print(cluster_dis)
cluster_index = np.argmin(cluster_distance)
print("该样本所属簇的标号为", cluster_index)#从0开始计算
# print(data.shape)
clf = joblib.load("model_for_custers/cluster_7SVR.model")
#1 cluster_1 RR     25.5       413.5841
#2 cluster_1 RR   0.05         217.5486
#3 cluster_0 SVR    C=2200      283.1851
#4 cluster_0 SVR    C=2200      306.2867
#5 cluster_1 RR  0.02          690.9123
#6 cluster_7 SVR   C=0.1       379.8833
#7 cluster_0 SVR C=2200   218.4674
#8 cluster_0 SVR C=3000       118.0444
# df2 = pd.read_excel("data_files/多尺度样本聚类_912.xls",
#                              sheetname="cluster_"+str(cluster_index))
# cluster_data = df2.values[:, 0:27]
# cluster_min  = np.min(cluster_data, axis=0)#g该簇上最小值
# cluster_max = np.max(cluster_data, axis=0)#该簇上最大值
# print(c)
# print("22222", target[0:2])
# norm_data, norm_target = (validation_dataX[6:7]-cluster_min)/(cluster_max-cluster_min), \
#                          validation_targetY[6:7]
print("true value", np.exp(target))
print(np.exp(target), np.exp(clf.predict(data)))
#1 #RR 251 SVR 228.4340

#1  cluster_2 RR 754.79
#2  cluster_8 MLR 212.6244