import numpy as np
import pandas as pd
import xlrd


def load_data(input_file):
    source_data = pd.read_excel(input_file)
    np_data = source_data.source_data.iloc[:,:].values
    # print(source_data.columns)
    return source_data, np_data


def data_process(np_data):
    from sklearn import preprocessing
    array_data = np.zeros(np_data.shape)
    for i in range(26):
        # array_data[:, i+1] = preprocessing.minmax_scale(np_data[:, i+1])
        array_data[:, i] = preprocessing.minmax_scale(np_data[:, i])
    # array_data[:, 23] = np.log(np_data[:, 23])
    # print(array_data.shape)
    return array_data


def cluster_split(array_data):
    print(array_data[1, 1:-1])
    return array_data[:, 1:-1]


def train_cluster(train_data, array_data, source_data):
    from sklearn.cluster import KMeans, DBSCAN
    # model = DBSCAN()
    model = KMeans()
    model.fit(train_data)
    label = np.zeros((len(model.labels_), 1), dtype=int)
    for i in range(len(model.labels_)):
        label[i, 0] = int(model.labels_[i])
    # print(train_data, model.cluster_centers_)
    # r = pd.concat([source_data, pd.Series(model.labels_, index=source_data.index)], axis=1)

    # print(labels)
    combine = np.concatenate((array_data, label), axis=1)
    writer = pd.ExcelWriter('cluster_data2.xls')
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
    plot_cluster(train_data, model.labels_)
    # r2 = pd.concat([pd.DataFrame(model.cluster_centers_)])
    # r2.columns = source_data.columns[1:]
    # r2.to_excel(writer, sheet_name='center')
    # writer.save()
    # r2 = pd.concat([array_data, model.labels_], axis=0)
    # print(combine[1, :])
    # r1 = pd.DataFrame(list(model.labels_))
    # r2 = pd.concat([r1])
    # r2.to_excel(writer, sheet_name='name2')
    writer.save()


def plot_cluster(data_zs, r):
    from sklearn.manifold import TSNE

    tsne = TSNE()
    tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    tsne = pd.DataFrame(tsne.embedding_)  # 转换数据格式

    import matplotlib.pyplot as plt

    d = tsne[r == 0]
    plt.plot(d[0], d[1], 'k.')

    d = tsne[r == 1]
    plt.plot(d[0], d[1], 'r.')

    d = tsne[r == 2]
    plt.plot(d[0], d[1], 'y.')
    d = tsne[r == 3]
    plt.plot(d[0], d[1], 'g.')
    d = tsne[r == 4]
    plt.plot(d[0], d[1], 'c.')

    d = tsne[r == 5]
    plt.plot(d[0], d[1], 'm.')
    d = tsne[r == 6]
    plt.plot(d[0], d[1], 'b.')
    d = tsne[r == 7]
    plt.plot(d[0], d[1], '#EE82EE',marker='.',linestyle='dotted')

    plt.show()

def run_cluster():
    print("run_cluster")
    resource_data, np_data = load_data('多尺度样本_revision.xlsx')
    array_data = data_process(np_data)
    train_data = cluster_split(array_data)
    # print(array_data, np_data)
    train_cluster(train_data, np_data, resource_data)


if __name__ == "__main__":
    print('welcome to cluster world')
    run_cluster()
