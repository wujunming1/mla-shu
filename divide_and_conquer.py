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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
def load_data(filename):
    file_list = filename.strip().split(".")
    data_frame = None
    if file_list[-1] == "xlsx":
        data_frame = pd.read_excel(filename)
    elif file_list[-1] == "csv":
        data_frame = pd.read_csv(filename)
    else:
        data_frame = None
    input_data = data_frame.values
    n_samples, n_dimension = input_data.shape
    print("样本数为%d，维度为%d" % (n_samples, n_dimension))
    feature_names = [column for column in data_frame]#列属性名
    print(feature_names)
    return input_data


def data_precessing(data_set):
    #对原始数据进行标准差归一化
    dataX, dataY = data_set[:, 1:-1], data_set[:, -1]
    Normalize_dataX = MinMaxScaler().fit_transform(dataX)
    # Normalize_dataX = StandardScaler().fit_transform(dataX)
    Normalize_dataY = np.log(dataY)
    return Normalize_dataX, Normalize_dataY


# def gaussian_model():
#     #GPR model
#     from sklearn.gaussian_process import GaussianProcessRegressor
#     from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
#     from sklearn.gaussian_process.kernels import ConstantKernel as C
#     # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
#     parameter = ' gaussian_model '
#     # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
#     kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000))
#     model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
#     return model, parameter


def k_means_cluster(train_data):
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(train_data)
    labels = kmeans.labels_#聚类标签
    clusters_center = kmeans.cluster_centers_
    return labels, clusters_center


def train_cluster(train_data, array_data, source_data):
    from sklearn.cluster import KMeans
    model = KMeans()
    model.fit(train_data)
    label = np.zeros((len(model.labels_), 1), dtype=int)
    for i in range(len(model.labels_)):
        label[i, 0] = int(model.labels_[i])
    # print(train_data, model.cluster_centers_)
    # r = pd.concat([source_data, pd.Series(model.labels_, index=source_data.index)], axis=1)

    # print(labels)
    combine = np.concatenate((array_data, label), axis=1)
    writer = pd.ExcelWriter('brg\\cluster_data2.xls')
    r0 = pd.concat([pd.DataFrame(array_data[:, 0:2]), pd.DataFrame(model.labels_)], axis=1)
    # r0.columns = ['temp', 'stress','stacking','DL','G','L','Ni3Al', 'label']
    r0.columns = ['temp', 'stress', 'label']
    r0.to_excel(writer, sheet_name='cluster_label')
    for i in range(len(np.unique(model.labels_))):
        cluster_subset = combine[combine[:, -1] == i][:, :-1]
        # print(np.arange(0, len(cluster_subset[:, 0])+1, 1).T)
        r0 = pd.DataFrame(np.arange(0, int(len(cluster_subset[:, 0])), 1).T)
        r1 = pd.DataFrame(cluster_subset)
        r = pd.concat([r0, r1], axis=1)
        r.columns = ['alloy'] + list(source_data.columns)
        r.to_excel(writer, sheet_name='cluster_'+str(i))
    writer.save()


def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True


    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


# 定义要创建的目录
def linear_prediction(data, target):
    from sklearn.linear_model import LinearRegression
    parameter = ' Linear_model '
    linear_model = LinearRegression()
    linear_model.fit(data, target)
    pre_y = linear_model.predict(data)
    rmse = np.sqrt(mean_squared_error(target, pre_y))
    return linear_model, rmse


def lasso_regression(data, target):
    from sklearn.linear_model import Ridge, Lasso
    parameter = 'Lasso_model'
    lasso_model = Lasso(alpha=0.01) #alpha = 0.1,,0.5,1.0
    lasso_model.fit(data, target)
    pre_y = lasso_model.predict(data)
    rmse = np.sqrt(mean_squared_error(target, pre_y))
    return lasso_model, rmse


def ridge_prediction(data, target):
    from sklearn.linear_model import Ridge
    parameter = ' Elastic_model '
    ridge_model = Ridge(alpha=0.5)

    ridge_model.fit(data, target)
    pre_y = ridge_model.predict(data)
    rmse = np.sqrt(mean_squared_error(target, pre_y))
    return ridge_model, rmse


def gaussian_prediction(data, target):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.metrics import mean_squared_error
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter_1 = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 1000))
    gaussian_model = GaussianProcessRegressor(kernel=kernel, alpha=0.001, n_restarts_optimizer=20)


    gaussian_model.fit(data, target)
    pre_y = gaussian_model.predict(data)
    rmse = np.sqrt(mean_squared_error(target, pre_y))
    return gaussian_model, rmse


def svr_prediction(data, target):
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    parameter_2 = ' svr_model '
    svr_model = SVR(kernel='rbf', C=800, gamma='auto')

    svr_model.fit(data, target)
    pre_y = svr_model.predict(data)
    rmse = np.sqrt(mean_squared_error(pre_y,  target))
    return svr_model, rmse


def rf_prediction(data, target):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    parameter_3 = ' RandomForest_model '
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    random_forest_model = RandomForestRegressor(n_estimators=15, max_depth=8, criterion='mae', bootstrap=True)

    random_forest_model.fit(data,target)
    pre_y = random_forest_model.predict(data)
    rmse = np.sqrt(mean_squared_error(pre_y, target))

    return random_forest_model, rmse


def mape_function(y_pred, y_true):
    return abs(np.sum((y_pred - y_true) / y_true) / len(y_pred))


def plot_cluster(train_data, cluster_label, subplot_num):
    #聚类结果的可视化
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    trans_data = pca.fit_transform(train_data)
    print(trans_data)
    plt.subplot(subplot_num)
    for row_index in range(len(trans_data)):
        row_data = trans_data[row_index]
        if(cluster_label[row_index]==0):
            plt.scatter(row_data[0], row_data[1], c='#FF00FF', s=8, label="Cluster1")	#洋红#FF00FF #cluster1
        elif(cluster_label[row_index]==1):
            plt.scatter(row_data[0], row_data[1],c='r', s=8, label="Cluster2")#红色 #cluster2
        elif (cluster_label[row_index] == 2):
            plt.scatter(row_data[0], row_data[1], c='g', s=8, label="Cluster3")#绿色 cluster3
        elif (cluster_label[row_index] == 3):
            plt.scatter(row_data[0], row_data[1],c='b', s=8, label="Cluster4")#蓝色cluster4
        elif (cluster_label[row_index] == 4):
            plt.scatter(row_data[0], row_data[1],c='#696969', s=8, label="Cluster5")#暗灰色 cluster5
        # elif (cluster_label[row_index] == 5):
        #     plt.scatter(row_data[0], row_data[1], c='#FFA500')#橙色 cluster6
        # elif (cluster_label[row_index] == 6):
        #     plt.scatter(row_data[0], row_data[1], c='#00BFFF')#	深天蓝#00BFFF cluster7
        else:
            plt.scatter(row_data[0], row_data[1], c='#7CFC00', s=8, label="Cluster6")#草坪绿 cluster8
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.xticks(trans_data[:, 0])
    # plt.xticks(trans_data[:, 1])
    return plt


from sklearn.model_selection import KFold
input_data = pd.read_excel("data_files/多尺度样本_revision.xlsx")
print(input_data)
input_data = input_data.values
#读取274条合金样本
print(input_data.shape)
normalize_dataX, normalize_dataY = data_precessing(input_data)
#对数据进行归一化处理
kf = KFold(n_splits=10)
fold_count = 0
all_pred_y = []
rmse_sum = 0
mape_sum = 0
rmse_all = []
mape_all = []
plt = None
for train_index, test_index in kf.split(input_data):
    train_x, train_y = normalize_dataX[train_index],normalize_dataY[train_index]
    test_x, test_y = normalize_dataX[test_index], normalize_dataY[test_index]
    cluster_labels, cluster_centers = k_means_cluster(train_x)
    print("ddddd", cluster_labels , cluster_centers)
    # plw = plot_cluster(train_x, cluster_labels, int("33"+str(fold_count+1)))
    clu_labels = np.array(cluster_labels)
    print(clu_labels)
    #每择（10次）交叉验证，训练集都聚成6个簇
    optimal_models = []
    for index in range(8):
        #为每个簇选择最优回归预测模型
        cluster_index = np.argwhere(clu_labels==index)
        cluster_index = cluster_index[:, 0]
        cluster_data_x = train_x[cluster_index]#每个簇中样本的特征或者输入
        cluster_data_y = train_y[cluster_index]#每个簇中样本的目标属性或者输出
        print("1111", cluster_data_x, cluster_data_y)
        gaussian_model, gaussian_rmse = gaussian_prediction(cluster_data_x ,cluster_data_y)
        svr_model, svr_rmse = svr_prediction(cluster_data_x, cluster_data_y)
        rf_model, rf_rmse = rf_prediction(cluster_data_x, cluster_data_y)
        # linear_model, mlr_rmse = linear_prediction(cluster_data_x, cluster_data_y)
        lasso_model, lasso_rmse = lasso_regression(cluster_data_x, cluster_data_y)
        ridge_model, ridge_rmse = ridge_prediction(cluster_data_x, cluster_data_y)
        print("dddaff1ff", rf_model)
        models = ["gaussian_model", "svr_model", "random_forest_model","lasso_model", "ridge_model"]

        rmses = [gaussian_rmse, svr_rmse, rf_rmse,
                 lasso_rmse, ridge_rmse]
        print("各模型的rmse为", rmses)
        print("1111111ddddd", rmses)
        optimal_cluster_model = models[rmses.index(min(rmses))]
        print("最优回归模型", optimal_cluster_model)
        print("111111", gaussian_model)
        # print(optimal_cluster_model==gaussian_model)
        import pickle
        from sklearn.externals import joblib
        from sklearn.preprocessing import MinMaxScaler

        mkpath = str(fold_count)+"model_for_custers/"
        # 调用函数
        mkdir(mkpath)
        if(optimal_cluster_model == "gaussian_model"):
            gaussian_model.fit(cluster_data_x,cluster_data_y)
            joblib.dump(gaussian_model, str(fold_count)+"model_for_custers/" + "cluster"+str(index) + "GPR" + ".model")
        elif(optimal_cluster_model == "svr_model"):
            svr_model.fit(cluster_data_x, cluster_data_y)
            joblib.dump(svr_model, str(fold_count)+"model_for_custers/" + "cluster"+str(index) + "SVR" + ".model")
        # elif(optimal_cluster_model=="mlr_model"):
        #     linear_model.fit(cluster_data_x, cluster_data_y)
        #     joblib.dump(linear_model, str(fold_count) + "model_for_custers/" + "cluster" + str(index) + "MLR" + ".model")
        elif (optimal_cluster_model == "lasso_model"):
            lasso_model.fit(cluster_data_x, cluster_data_y)
            joblib.dump(lasso_model,
                        str(fold_count) + "model_for_custers/" + "cluster" + str(index) + "LR" + ".model")
        elif (optimal_cluster_model == "ridge_model"):
            ridge_model.fit(cluster_data_x, cluster_data_y)
            joblib.dump(ridge_model,
                        str(fold_count) + "model_for_custers/" + "cluster" + str(index) + "RR" + ".model")
        else:
            rf_model.fit(cluster_data_x, cluster_data_y)
            joblib.dump(rf_model, str(fold_count)+"model_for_custers/" + "cluster"+str(index) + "RF" + ".model")
        optimal_models.append(optimal_cluster_model)
    print("all models have already save!")
    test_indice = 0
    y_test = []
    for test_x_index in test_x:
        #判断测试集中每条样本属于哪个簇
        # min_distance = np.linalg.norm(test_x[0]-cluster_centers[0])
        # # belongto_cluster = 0
        # for center_ in cluster_centers:
        #     cluster_distance = np.linalg.norm(test_x_index, center_)
        #     if(cluster_distance<=min_distance):
        #         min_distance = cluster_distance
        #         # belongto_cluster+=1

        distances = []
        for center in cluster_centers:
            center_1 = np.array(center)
            test_x_index_1 = np.array(test_x_index)
            # print("簇中心", center)
            # print("测试样本", test_x_index)
            distances.append(np.linalg.norm(test_x_index_1- center_1))
        min_cluster_index = distances.index(min(distances))
        predict_model=optimal_models[min_cluster_index]
        if predict_model=="gaussian_model":
            clf = joblib.load(str(fold_count)+"model_for_custers/" + "cluster"+str(min_cluster_index) + "GPR" + ".model")
            # print("22222", target)
            print(clf.predict([test_x_index]), test_y[test_indice])
            y_test.append(clf.predict([test_x_index])[0])
        elif predict_model=="svr_model":
            clf = joblib.load(str(fold_count) + "model_for_custers/" + "cluster" + str(min_cluster_index) + "SVR" + ".model")
            # print("22222", target)
            print(clf.predict([test_x_index]), test_y[test_indice])
            y_test.append(clf.predict([test_x_index])[0])
        elif predict_model=="lasso_model":
            clf = joblib.load(str(fold_count) + "model_for_custers/" + "cluster" + str(min_cluster_index) + "LR" + ".model")
            # print("22222", target)
            print(clf.predict([test_x_index]), test_y[test_indice])
            y_test.append(clf.predict([test_x_index])[0])
        # elif predict_model == "mlr_model":
        #     clf = joblib.load(
        #         str(fold_count) + "model_for_custers/" + "cluster" + str(min_cluster_index) + "MLR" + ".model")
        #     # print("22222", target)
        #     print(clf.predict([test_x_index]), test_y[test_indice])
        #     y_test.append(clf.predict([test_x_index])[0])
        elif predict_model == "ridge_model":
            clf = joblib.load(
                str(fold_count) + "model_for_custers/" + "cluster" + str(min_cluster_index) + "RR" + ".model")
            # print("22222", target)
            print(clf.predict([test_x_index]), test_y[test_indice])
            y_test.append(clf.predict([test_x_index])[0])
        else:
            clf = joblib.load(str(fold_count) + "model_for_custers/" + "cluster" + str(min_cluster_index) + "RF" + ".model")
            # print("22222", target)
            print(clf.predict([test_x_index]), test_y[test_indice])
            y_test.append(clf.predict([test_x_index])[0])
        test_indice+=1
    fold_count += 1
    all_pred_y.extend(y_test)
    rmse = np.sqrt(mean_squared_error(test_y, y_test))
    ab_error = 0
    for i in range(len(test_y)):
        diff = test_y[i]-y_test[i]
        absolute_error = abs(diff/test_y[i])
        ab_error += absolute_error
    mape = ab_error/len(test_y)
    rmse_sum += rmse
    mape_sum += mape
    rmse_all.append(rmse)
    mape_all.append(mape)
# plw.show()
print("模型预测值为", all_pred_y, "模型真实值为",normalize_dataY)
df1_rmse = pd.Series(rmse_all)
df2_mape = pd.Series(mape_all)
print(df1_rmse.describe())
print(df2_mape.describe())
rmse = df1_rmse.describe()["min"]
mape  = df2_mape.describe()["min"]
true_pred = list(zip(normalize_dataY, all_pred_y))
df = pd.DataFrame(true_pred, columns=["真实值", "预测值"])
df.to_excel("蠕变断裂寿命真实值和预测值.xlsx")
df = pd.read_excel("蠕变断裂寿命真实值和预测值.xlsx")
predicted_results = df.values
mean_rmse = np.sqrt(mean_squared_error(predicted_results[:, 0], predicted_results[:, 1]))
mean_mape = mape_function(predicted_results[:, 0], predicted_results[:, 1])
r2_score1 = r2_score(predicted_results[:, 0], predicted_results[:, 1])
print("模型的平均预测精度rmse为", rmse, "模型的平均预测精度mape为", mape,
      "r2分数为", r2_score1)



