import pandas as pd

df = pd.read_csv('./data/madelon_train.data', sep=' ', header=None).drop(500, axis=1)

df_labels = pd.read_csv('./data/madelon_train.labels', sep=' ', header=None)

df_test = pd.read_csv('./data/madelon_valid.data', sep=' ', header=None).drop(500, axis=1)

df_test_labels = pd.read_csv('./data/madelon_valid.labels', sep=' ', header=None)

y_train = df_labels[0]
X_train = df
y_test = df_test_labels[0]
X_test = df_test

X_tr_1 = X_train[:600]
y_tr_1 = y_train[:600]

X_tr_2 = X_train[600:1200]
y_tr_2 = y_train[600:1200]

X_tr_3 = X_train[1200:1800]
y_tr_3 = y_train[1200:1800]


top_corrs = [28,
 48,
 64,
 105,
 128,
 153,
 241,
 281,
 318,
 336,
 338,
 378,
 433,
 442,
 451,
 453,
 455,
 472,
 475,
 493]