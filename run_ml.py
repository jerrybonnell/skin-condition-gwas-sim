import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import seaborn as sns
import umap
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


assert len(sys.argv) == 3
print(sys.argv)
CONFIG = int(sys.argv[1])
COHORT = sys.argv[2]

if COHORT == "hapmap":
    df = pd.read_hdf("recoded_hapmap_ped_case_control.h5")
    df = df.iloc[:, 6:]
    start_index = 6
    known_missing = ['rs4379175', 'rs13437088',
                      'rs582757', 'rs10738626', 'rs11652075', 'rs545979']
    with open("gen_info/col_map.pkl", "rb") as f:
        col_map = pickle.load(f)
    with open("gen_info/sig.info", "r") as f:
        sig = list(set(
            [col_map[int(l.strip().split(" ")[0])] for l in f.readlines()]))
elif COHORT == "1KG":
    df = pd.read_hdf("recoded_hapmap_ped_case_control_vcf.h5")
    start_index = 0
    _ = df.pop("index")
    known_missing = ['rs10865331',
                     'rs2111485',
                     'rs1295685',
                     'rs848',
                     'rs9504361',
                     'rs11795343',
                     'rs4561177',
                     'rs8016947',
                     'rs2041733',
                     'rs16948048',
                     'rs11652075']
    with open("gen_info_vcf/sig.info", "r") as f:
        sig = list(set([l.strip().split(" ")[0] for l in f.readlines()]))
elif COHORT == "together":
    df = pd.read_hdf("together_hapmap_ped_case_control_supervised.h5")
    start_index = 0
    _ = df.pop("dataset")
    known_missing = ['rs10865331', 'rs2111485', 'rs1295685', 'rs848', 'rs847', 'rs4379175', 'rs12188300', 'rs9504361', 'rs13437088', 'rs582757', 'rs10738626', 'rs11795343', 'rs4520482', 'rs1250546', 'rs645078', 'rs4561177', 'rs8016947', 'rs1665050', 'rs2041733', 'rs16948048', 'rs11652075', 'rs545979', 'rs1056198', 'rs10994675']
    with open("together_sig.pkl", "rb") as f:
        sig = pickle.load(f)
else:
    raise ValueError("unk")
print((len(df), len(df.columns)))

to_drop = []
df_cols = list(df.columns)
for i in tqdm(range(start_index, len(df_cols))):
    #3 in set(recoded_df[i].value_counts().index):
    if df[df_cols[i]].isna().sum().sum() > 0:  # or \
        to_drop.append(df_cols[i])
assert len(to_drop) == len(known_missing)
for snp in to_drop:
    assert snp in known_missing


df_y = df.pop('group')
remainder = list(set(list(df.columns)) - set(sig))
assert len(sig) == len(set(sig))
print(len(sig))
# TODO toggle
# np.random.seed(2022)
if CONFIG == 0:
    # list(np.random.choice(remainder, 10000, replace=False))
    column_set = sig + remainder
elif CONFIG == 1:
    column_set = sig
elif CONFIG == 2:
    column_set = list(np.random.choice(remainder, 10000, replace=False))
else:
    assert 1 == 2
assert column_set is not None
# column_set = list(np.random.choice(remainder, 10000))
# column_set = [int(c) for c in column_set]
print(CONFIG)
print(len(column_set))

df_X = df[column_set]

assert len([l for l in list(zip(df_X.columns, df_X.dtypes.tolist()))
 if l[1] != np.float64 and l[1] != np.int64]) == 0

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
models = [
    ('LASSO', LogisticRegression(solver='liblinear', penalty='l1')),
    ('RIDGE', LogisticRegression(solver='liblinear', penalty='l2')),
    ('RF',  RandomForestClassifier()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('15-NN', KNeighborsClassifier(n_neighbors=15)),
    ('SVM-LINEAR', SVC(kernel='linear', probability=True)),
    ('SVM-RBF', SVC(kernel='rbf', probability=True)),
    ('MLP', MLPClassifier(solver='lbfgs'))
]

colors = plt.get_cmap('jet')(np.linspace(0, 1, len(models)))

model2tprs = defaultdict(list)
model2aucs = defaultdict(list)
model2metrics = defaultdict(list)
mean_fpr = np.linspace(0, 1, 100)

# X = df_X.to_numpy()
# y = df_y.to_numpy()

le = LabelEncoder()
y = le.fit_transform(df_y)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

model2pred = {}
model2proba = {}
for name, model in models:
    model2pred[name] = np.zeros(len(y))
    model2proba[name] = np.zeros(len(y))


model2comp = {}
for model in ['SVM-LINEAR', 'LASSO', 'RIDGE']:
    model2comp[model] = np.zeros((10, len(df_X.columns)))
# summed_comp_scores = np.zeros((10, len(df_X.columns)))
cv_index = 0
for train_index, test_index in tqdm(cv.split(df_X, y), total=10):
    X_train_transf = df_X.iloc[train_index, :].copy()
    X_test_transf = df_X.iloc[test_index, :].copy()
    y_train = y[train_index]
    y_test = y[test_index]

    col2mode = {}
    for index in known_missing:
        # index = col_map[int_index]
        col2mode[index] = X_train_transf[index].mode().iloc[0]
        X_train_transf[index] = X_train_transf[index].fillna(col2mode[index])
        X_test_transf[index] = X_test_transf[index].fillna(col2mode[index])

    sel = VarianceThreshold()
    sel.fit(X_train_transf)
    X_train_transf = X_train_transf.loc[:, sel.get_support()]
    X_test_transf = sel.transform(X_test_transf)
    if cv_index == 0:
        print(len(X_train_transf.columns))
    # assert 1==2

    included = [col in sig for col in list(X_train_transf.columns)]
    included_pos = []
    col_list = list(X_train_transf.columns)
    for i in range(len(col_list)):
        if col_list[i] in sig:
            included_pos.append(i)
    # print(len(included_pos))
    # print(len(included))
    print(f"sig vars dropped : {len(sig) - sum(included)}")
    print(
        f"features removed : {len(column_set) - len(X_train_transf.columns)}")
    print(f"remaining : {len(X_train_transf.columns)}")

    for name, model in models:
        probas_ = model.fit(
            X_train_transf, y_train).predict_proba(X_test_transf)
        # print(model.classes_)
        # assert 1 == 2
        probas_true = [p[0] for p in probas_]
        # Compute ROC curve and area the curve
        y_pred = model.predict(X_test_transf)
        model2proba[name][test_index] = probas_true
        model2pred[name][test_index] = y_pred

        if name in ["SVM-LINEAR", "LASSO", "RIDGE"]:
            int_mask = sel.get_support(indices=True)
            # int_mask = list(range(len(X_train_transf.columns)))
            #my_coefs = model.coef_
            # print(np.abs(model.coef_[0]))
            model2comp[name][cv_index, int_mask] = np.abs(model.coef_[0])
            # print(summed_comp_scores)
            # print("---")
    cv_index += 1

os.makedirs(f"model_{COHORT}", exist_ok=True)
ml_outs = {
    "models": models,
    "model2tprs": model2tprs,
    "model2aucs": model2aucs,
    "model2metrics": model2metrics,
    "mean_fpr": mean_fpr,
    "model2pred": model2pred,
    "model2proba": model2proba,
    "model2comp": model2comp,
    "df_X": df_X,
    "df_y": df_y,
    "y": y,
    "sig": sig
}
with open(f"model_{COHORT}/ml_{COHORT}_out.pkl", "wb") as f:
    pickle.dump(ml_outs, f)




