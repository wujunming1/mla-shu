import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
# def mscatter(x,y,ax=None, m=None, **kw):
#     import matplotlib.markers as mmarkers
#     if not ax: ax=plt.gca()
#     sc = ax.scatter(x,y,**kw)
#     if (m is not None) and (len(m)==len(x)):
#         paths = []
#         for marker in m:
#             if isinstance(marker, mmarkers.MarkerStyle):
#                 marker_obj = marker
#             else:
#                 marker_obj = mmarkers.MarkerStyle(marker)
#             path = marker_obj.get_path().transformed(
#                         marker_obj.get_transform())
#             paths.append(path)
#         sc.set_paths(paths)
#     return sc
df = pd.read_excel("data_files/多尺度样本_聚类标签.xlsx")
data_array = df.values
train_data, label = data_array[:, 0:28], data_array[:, -1]
train_data = MinMaxScaler().fit_transform(train_data)
pca = PCA(n_components=2)
trans_data = pca.fit_transform(train_data)
print(trans_data)
print(pca.explained_variance_ratio_)
new_data = np.column_stack((trans_data, label))
print("data of transformation is", new_data)
df = pd.DataFrame(new_data)
df.to_excel("ClusterVisualation.xlsx")
for row_index in range(len(trans_data)):
    row_data = trans_data[row_index]
    if(label[row_index]==0):
        plt.scatter(row_data[0], row_data[1], c='#FF00FF')	#洋红#FF00FF #cluster1
    elif(label[row_index]==1):
        plt.scatter(row_data[0], row_data[1],c='r')#红色 #cluster2
    elif (label[row_index] == 2):
        plt.scatter(row_data[0], row_data[1],c='g')#绿色 cluster3
    elif (label[row_index] == 3):
        plt.scatter(row_data[0], row_data[1],c='b')#蓝色cluster4
    elif (label[row_index] == 4):
        plt.scatter(row_data[0], row_data[1],c='#696969')#暗灰色 cluster5
    elif (label[row_index] == 5):
        plt.scatter(row_data[0], row_data[1], c='#FFA500')#橙色 cluster6
    elif (label[row_index] == 6):
        plt.scatter(row_data[0], row_data[1], c='#00BFFF')#	深天蓝#00BFFF cluster7
    else:
        plt.scatter(row_data[0], row_data[1], c='#7CFC00')#草坪绿 cluster8
plt.show()