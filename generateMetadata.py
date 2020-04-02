import pandas as pd
import time
import os
import xlwt
import xlrd
import xlutils.copy
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.cluster import estimate_bandwidth, KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, \
    SpectralClustering, AgglomerativeClustering, FeatureAgglomeration, DBSCAN, Birch, OPTICS
from processing import data_missing, data_processing
from evaluation import hopkins, sil_score

def traverse(f):
    global filename
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            filename.append(tmp_path)
        else:
            traverse(tmp_path)

def hyperopt_train(params):
    global X
    algorithm = params['type']
    del params['type']
    if algorithm == 0:
        clf = KMeans(**params)
    elif algorithm == 1:
        clf = MiniBatchKMeans(**params)
    elif algorithm == 2:
        clf = AffinityPropagation(**params)
    elif algorithm == 3:
        clf = MeanShift(**params)
    elif algorithm == 4:
        clf = SpectralClustering(**params)
    elif algorithm == 5:
        if params['linkage'] == 'ward':
            params['affinity'] = 'euclidean'
        clf = AgglomerativeClustering(**params)
    elif algorithm == 6:
        if params['linkage'] == 'ward':
            params['affinity'] = 'euclidean'
        clf = FeatureAgglomeration(**params)
    elif algorithm == 7:
        clf = DBSCAN(**params)
    elif algorithm == 8:
        clf = OPTICS(**params)
    elif algorithm == 9:
        clf = Birch(**params)
    else:
        return 0
    try:
        clf.fit(X)
        score = sil_score(X, clf.labels_)
    except Exception as e:
        print(e)
        score = -1
    return score


def f(params):
    global best_score, best_time, best_params, early_stopping
    score = hyperopt_train(params)
    if score > best_score:
        print('score: ', score, 'using ', params)
        best_params = params
        best_score = score
    return {'loss': -score, 'status': STATUS_OK}


if __name__ == "__main__":

    filename = []
    traverse("../data/")
    print(filename)

    # workbook = xlwt.Workbook()
    # worksheet_score = workbook.add_sheet('score')
    # worksheet_params = workbook.add_sheet('params')
    # workbook.save("result.xls")

    try:
        for col in range(len(filename)):
            print(filename[col])
            excel_data = xlrd.open_workbook('result.xls')
            ws = xlutils.copy.copy(excel_data)
            worksheet_score = ws.get_sheet(0)
            worksheet_params = ws.get_sheet(1)
            worksheet_score.write(col, 0, filename[col])
            worksheet_params.write(col, 0, filename[col])

            ####文件讀取
            source_data = pd.read_excel(filename[col])
            source_data.dropna(inplace=True)
            np_data = source_data.values
            np_data = data_missing(np_data)

            if np_data.shape[0] > 5000 or np_data.shape[1] > 51:
                continue

            ###条件属性与决策属性
            X = np_data[:, 0:np_data.shape[1]-1]  ###条件属性
            y = np_data[:, np_data.shape[1] - 1]

            ###对X、y进行预处理 X去空值与独热编码、归一化 y数值编码
            X = data_processing(X)
            if X.shape[1] > 300:
                continue
            # try:
            #     result = hopkins(X)
            # except Exception as e:
            #     print(e)
            #     result = 0
            # if result < 0.7:
            #     continue

            ###定义参数空间
            kmeans = {'n_clusters': hp.choice('n_clusters', range(2, 30)),
                      'algorithm': hp.choice('algorithm', ['auto', 'full', 'elkan']),
                      'init': hp.choice('init', ['k-means++', 'random']),
                      'precompute_distances': hp.choice('precompute_distances', ['auto', True, False]),
                      'tol': hp.uniform('tol', 1e-5, 1e-2),
                      'type': 0}
            minbatchkmeans = {'n_clusters': hp.choice('n_clusters', range(2, 30)),
                              'batch_size': hp.choice('batch_size', range(20, 200)),
                              'max_iter': 300,
                              'tol': hp.uniform('tol', 1e-5, 1e-2),
                              'reassignment_ratio': hp.uniform('reassignment_ratio', 1e-3, 1e-1),
                              'type': 1}
            affnitypropagation = {'damping': hp.uniform('damping', 0.5, 1),
                                  'convergence_iter': hp.choice('convergence_iter', range(10, 50)),
                                  'affinity': hp.choice('affinity', ['precomputed', 'euclidean']),
                                  'type': 2}
            bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=len(X))
            meanshift = {'cluster_all': hp.choice('cluster_all', [True, False]),
                         'min_bin_freq': hp.choice('min_bin_freq', range(1, 10)),
                         'bin_seeding': hp.choice('bin_seeding', [True, False]),
                         'bandwidth': hp.uniform('bandwidth', bandwidth - bandwidth / 2, bandwidth + bandwidth / 2),
                         'type': 3}
            spectralclustering = {'n_clusters': hp.choice('n_clusters', range(2, 30)),
                                  'eigen_solver': hp.choice('eigen_solver', [None, 'arpack', 'lobpcg', 'amg']),
                                  'n_init': hp.choice('n_init', range(2, 20)),
                                  'gamma': hp.uniform('gamma', 0, 10),
                                  'assign_labels': hp.choice('assign_labels', ['kmeans']),
                                  'degree': hp.choice('degree', range(2, 7)),
                                  'coef0': hp.uniform('coef0', 0, 2),
                                  'affinity': hp.choice('affinity', ['nearest_neighbors', 'rbf']),
                                  'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
                                  'type': 4}
            agglomerativelustering = {'n_clusters': hp.choice('n_clusters', range(2, 30)),
                                      'affinity': hp.choice('affinity', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']),
                                      'linkage': hp.choice('linkage', ['ward', 'complete', 'average', 'single']),
                                      'compute_full_tree': hp.choice('compute_full_tree', ['auto', True, False]),
                                      'type': 5}
            featureagglomeration = {'n_clusters': hp.choice('n_clusters', range(2, 30)),
                                    'affinity': hp.choice('affinity', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']),
                                    'linkage': hp.choice('linkage', ['ward', 'complete', 'average', 'single']),
                                    'type': 6}
            dbsacn = {'eps': hp.uniform('eps', 0.1, 30),
                      'min_samples': hp.choice('min_samples', range(1, int(X.shape[0]/30))),
                      'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                      'leaf_size': hp.choice('leaf_size', range(1, 60)),
                      'type': 7}
            optics = {'min_samples': hp.choice('min_samples', range(2, int(X.shape[0]/30))),
                      'max_eps': hp.uniform('max_eps', 0.1, 30),
                      'p': hp.choice('p', range(1, 6)),
                      'min_cluster_size': hp.uniform('min_cluster_size', 0, 1),
                      'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                      'leaf_size': hp.choice('leaf_size', range(1, 60)),
                      'cluster_method': hp.choice('cluster_method', ['xi', 'dbscan']),
                      'eps': hp.uniform('eps', 0.1, 30),
                      'xi': hp.uniform('xi', 0, 1),
                      'type': 8}
            birch = {'threshold': hp.uniform('threshold', 0.1, 2),
                     'branching_factor': hp.choice('branching_factor', range(10, 200)),
                     'n_clusters': hp.choice('n_clusters', range(2, 30)),
                     'type': 9}
            for j in range(0, 10):
                best_score = -1
                best_params = {}
                trials = Trials()
                try:
                    if j == 0:
                        fmin(f, kmeans, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 1:
                        continue
                        # fmin(f, minbatchkmeans, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 2:
                        fmin(f, affnitypropagation, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 3:
                        fmin(f, meanshift, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 4:
                        continue
                        # fmin(f, spectralclustering, algo=tpe.suggest, max_evals=300, trials=trials)
                    elif j == 5:
                        fmin(f, agglomerativelustering, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 6:
                        continue
                        # fmin(f, featureagglomeration, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 7:
                        fmin(f, dbsacn, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 8:
                        fmin(f, optics, algo=tpe.suggest, max_evals=300, trials=trials)
                    elif j == 9:
                        fmin(f, birch, algo=tpe.suggest, max_evals=200, trials=trials)
                except Exception as e:
                    print(e)
                except Warning as w:
                    print(w)
                worksheet_score.write(col, j+1, best_score)
                worksheet_params.write(col, j+1, str(best_params))
            ws.save("result.xls")
    except Exception as e:
        print(e)



