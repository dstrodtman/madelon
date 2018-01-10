import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
from lib.helper import LikeFeatures
from lib.helper import con_cur_to_class_db
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
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
from sklearn.pipeline import Pipeline