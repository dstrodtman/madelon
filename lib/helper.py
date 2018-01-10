import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize, PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import boxcox
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import log_loss
# from tqdm import tqdm

def con_cur_to_class_db():
    con = pg2.connect(host='34.211.227.227',
                  dbname='postgres',
                  user='postgres')
    cur = con.cursor(cursor_factory=RealDictCursor)
    return con, cur

class LikeFeatures(object):
    
    def __init__(self, X, y):
        self.X = X.copy(deep=False)
        self.y = y
        self.corr_X = X.corr()
        self.Xy = X.copy(deep=False); self.Xy.insert(0, 'target', y)
        
    def find_corrs(self, corr_thresh = .5, count_thresh = 1):
        hi_corrs = self.corr_X.abs() > corr_thresh
        hi_count = self.corr_X[hi_corrs].count() > count_thresh
        self.top_corrs = list(self.corr_X[hi_count].index)
        self.top_corrs_mat = self.X[self.top_corrs].corr()
        self.top_corrs_target = self.top_corrs + ['target']
        self.top_corrs_target_mat = self.Xy[self.top_corrs_target].corr()

    def find_bins(self, corr_thresh = .95):
        if not self.top_corrs:
            self.find_corrs()
        corr_mask = self.top_corrs_mat[self.top_corrs_mat.abs() > corr_thresh].notnull()
        unique_bins = set(tuple(corr_mask.columns[corr_mask[col]]) \
                                for col in corr_mask.columns)
        self.feature_bins = dict(zip(range(len(unique_bins)), unique_bins))
    
    def find_best_bins(self, corr_thresh = .10, count_thresh = 7):
        if not self.feature_bins:
            self.find_bins()
        foot = [bins[0] for bins in self.feature_bins.values()]
        corr_foot = self.X[foot].corr()
        hi_corrs = corr_foot.abs() > corr_thresh
        lo_count = corr_foot[hi_corrs].count() < count_thresh
        self.best_bins = list(corr_foot[lo_count].index)
        
    def corr_heatmap(self, target=True, mask=False):
        fig = plt.figure(figsize=(20,15))
        if target:
            mat = self.top_corrs_target_mat
            target = 'target'
        else:
            mat = self.top_corrs_mat
            target = mat.columns[0]

        if mask:
            mask = np.zeros_like(mat)
            mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            ax = sns.heatmap(mat.sort_values(target, ascending=False)\
                             [list(mat.sort_values(target, ascending=False).index)],\
                             mask=mask,
                             annot = True, vmax=.3, square=True)
            
            
# class BestBins_log(object):
    
#     def __init__(self, LikeFeatures_object):
#         self.feature_bins = LikeFeatures_object.feature_bins
#         self.X = LikeFeatures_object.X
#         self.y = LikeFeatures_object.y
#         self.best_bins = []
#         self.first_features = [bins[0] for bins in self.feature_bins]
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        
#     def find_bins(self, degree=5):
#         results = {}

#         for combo in tqdm(combinations(self.first_features, degree):
#             X_train = self.X_train[list(combo)]
#             y_train = self.y_train

#             X_test = self.X_test[list(combo)]
#             y_test = self.y_test

#             X_tr_bc = pd.DataFrame()
#             X_ts_bc = pd.DataFrame()
#             for col in X_train.columns:
#                 box_cox_trans_tr, lmbda = boxcox(X_train[col]+.000001)
#                 box_cox_trans_ts = boxcox(X_test[col]+.000001, lmbda)
#                 X_tr_bc[col] = pd.Series(box_cox_trans_tr)
#                 X_ts_bc[col] = pd.Series(box_cox_trans_ts)

#             scaler = StandardScaler()
#             X_train_sc = scaler.fit_transform(X_tr_bc)
#             X_test_sc = scaler.transform(X_ts_bc)

#             results[combo_str] = {}

#             for k in range(4, 25):
#         #         trial += 1
#                 #     print('{}: {}'.format(trial, combo))

#                 if not results[combo_str]:
#                     results[combo_str]['best_r2'] = {}
#                     results[combo_str]['best_ll'] = {}
#                     results[combo_str]['best_r2']['k'] = 0
#                     results[combo_str]['best_r2']['r2'] = 0
#                     results[combo_str]['best_r2']['ll'] = 0
#                     results[combo_str]['best_ll']['k'] = 0
#                     results[combo_str]['best_ll']['r2'] = 0
#                     results[combo_str]['best_ll']['ll'] = 1

#                 KNC = KNeighborsClassifier(n_neighbors=k, weights='distance')
#                 KNC.fit(X_train_sc, y_train)
#                 temp_r2 = KNC.score(X_test_sc, y_test)
#                 temp_ll = log_loss(y_test, KNC.predict_proba(X_test_sc))

#                 if results[combo_str]['best_r2']['r2'] < temp_r2:
#                     results[combo_str]['best_r2']['k'] = k
#                     results[combo_str]['best_r2']['r2'] = temp_r2
#                     results[combo_str]['best_r2']['ll'] = temp_ll
#                 if results[combo_str]['best_ll']['ll'] > temp_ll:
#                     results[combo_str]['best_ll']['k'] = k
#                     results[combo_str]['best_ll']['r2'] = temp_r2
#                     results[combo_str]['best_ll']['ll'] = temp_ll
        