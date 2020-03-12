import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import datetime
import gc
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import json
from collections import OrderedDict
import warnings
import os
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 50)

a = pd.read_csv('MLB_CP_ITEMSEQ_SALES_DATA.csv')

# 필요한 column만 뽑아 합치기. 태블로에서는 미리 작업하니까 합치기만 하면 될 듯
a = a[['WDATE', '상권구분', 'ITEMSEQ', 'COLOR', 'AMT_ACT']]
a['ITEM'] = a.상권구분 + '+' + a.ITEMSEQ + '+' + a.COLOR
a.drop(['상권구분', 'ITEMSEQ', 'COLOR'], axis=1, inplace=True)
a = a.sort_values(['WDATE', 'ITEM'])

# test 기간 정하기. forecast_period 만큼.
te_start = '2019-06-27'
te_end = '2019-08-07'

# 예측하고 싶은 기간
forecast_period = (pd.to_datetime(te_end) - pd.to_datetime(te_start)).days + 1

# data의 마지막 구간. test set = holdout set
holdout = a[a['WDATE'] >= te_start]

# 매출 비중이 holdout set 에서 x% 이상
rev_rank = holdout.groupby('ITEM').apply(sum)['AMT_ACT'].reset_index()
rev_rank = rev_rank.sort_values('AMT_ACT', ascending=False).reset_index(drop=True)
rev_rank['rank'] = pd.DataFrame(np.array(["{0:03}".format(i) for i in range(1, len(rev_rank) + 1)]).reshape(-1, 1))
rev_rank['pct'] = round(rev_rank['AMT_ACT'] / sum(rev_rank['AMT_ACT']) * 100, 2)
rev_rank = rev_rank[rev_rank['pct'] >= 2]       # 여기가 x%
rev_rank['pct'] = rev_rank['pct'].astype('str')
rev_rank['list'] = rev_rank['rank'] + '_' + rev_rank['ITEM'] + '(' + rev_rank['pct'] + '%)'
rev_rank = rev_rank[['ITEM', 'list']]

num_category = len(rev_rank.list.unique())

a = pd.merge(a, rev_rank, how='inner', on='ITEM')
a.drop('ITEM', axis=1, inplace=True)
a.columns = ['WDATE', 'value', 'ITEM']

a = a.groupby(['WDATE', 'ITEM']).apply(sum)['value']
a = a.reset_index()

a.WDATE = pd.to_datetime(a.WDATE)

a = a.sort_values(['WDATE', 'ITEM'])

# 위까지 정리 다 됐음. 날짜, item, value(매출)
# 그림 한 번 보고
# for item in np.sort(a.ITEM.unique()):
#     plt.figure(figsize=(18, 5))
#     plt.plot(a[a.ITEM == item]['WDATE'], a[a.ITEM == item]['value'])
#     plt.title(item)
#     plt.show()

# 로그화를 할까말까
# a.value = np.log1p(a.value).fillna(0)

# scaling을 할까말까
# sc = a[a['WDATE'] < te_start]
# sc = sc.pivot('WDATE', 'ITEM', 'value')
# b = a.pivot('WDATE', 'ITEM', 'value')
# c = (b - sc.min()) / (sc.max() - sc.min())
# df_train = c.T.fillna(0)

# 그냥 가기
df_train = a
df_train = df_train.set_index(["ITEM", "WDATE"])[["value"]].unstack(level=-1).fillna(0)
df_train.columns = df_train.columns.get_level_values(1)

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus),
                            periods=periods, freq=freq)]


def prepare_dataset(df,
                    #promo_df,
                    t2017, is_train=True, name_prefix=None):
    X = pd.DataFrame(data=np.arange(0, len(rev_rank.list.unique())), columns=['item'])

    for i in [3, 5, 7, 10, 14, 21, 28]:
        tmp = get_timespan(df, t2017, i, i)
        # X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        # X['diff_%s_median' % i] = tmp.diff(axis=1).median(axis=1).values
        # X['diff_%s_min' % i] = tmp.diff(axis=1).min(axis=1).values
        # X['diff_%s_max' % i] = tmp.diff(axis=1).max(axis=1).values
        # X['diff_%s_std' % i] = tmp.diff(axis=1).std(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    # for i in [3, 5, 7, 10, 14, 21, 28]:
    #     tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)
    #     X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
    #     X['diff_%s_median_2' % i] = tmp.diff(axis=1).median(axis=1).values
    #     X['diff_%s_min_2' % i] = tmp.diff(axis=1).min(axis=1).values
    #     X['diff_%s_max_2' % i] = tmp.diff(axis=1).max(axis=1).values
    #     X['diff_%s_std_2' % i] = tmp.diff(axis=1).std(axis=1).values
    #     X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
    #     X['mean_%s_2' % i] = tmp.mean(axis=1).values
    #     X['median_%s_2' % i] = tmp.median(axis=1).values
    #     X['min_%s_2' % i] = tmp.min(axis=1).values
    #     X['max_%s_2' % i] = tmp.max(axis=1).values
    #     X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in range(1, 8):
        X['day_%s_lag' % i] = get_timespan(df, t2017, i, 1).values.ravel()
        X['diff_%s_lag' % i] = get_timespan(df, t2017, i + 1, i + 1).diff(axis=1, periods=i).iloc[:, -1].values

    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df, t2017, 4 * 7 - i, 4, freq='7D').mean(axis=1).values
        # X['mean_8_dow{}'.format(i)] = get_timespan(df, t2017, 8*7-i, 8, freq='7D').mean(axis=1).values
    #        X['mean_12_dow{}'.format(i)] = get_timespan(df, t2017, 12*7-i, 12, freq='7D').mean(axis=1).values
    #        X['mean_16_dow{}'.format(i)] = get_timespan(df, t2017, 16*7-i, 16, freq='7D').mean(axis=1).values
    #        X['mean_20_dow{}'.format(i)] = get_timespan(df, t2017, 20*7-i, 16, freq='7D').mean(axis=1).values


    # 날짜 정보
    # X['year'] = df[pd.date_range(t2017, periods=1)].columns.year[0]
    # X['month'] = df[pd.date_range(t2017, periods=1)].columns.month[0]
    # X['day'] = df[pd.date_range(t2017, periods=1)].columns.day[0]
    # X['dow'] = df[pd.date_range(t2017, periods=1)].columns.weekday[0]
    # X['weekend'] = (X['dow'] > 4).astype(int)

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=1)
        ].values
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


pred_start = date(2019, 6, 27)
bo_forecast_period = forecast_period   # 예측기간만큼
bo_pred_start = pred_start - timedelta(days=bo_forecast_period)
val_period = 1
num_days = 546              # training period
cat_feat = ['item',
            # 'year', 'month', 'dow', 'day', 'weekend'
            ]
init_points = 5            # 조사 시작점 randomly 몇 번?
n_iter = 15                 # init_points 중에 젤 나은 거에서 추가로 더 찾는거



def lgbm_bo(df_train, random_state=37):
    df_train_bo = df_train.copy()

    tr_start = bo_pred_start - timedelta(days=num_days + val_period)

    X_li, y_li = [], []

    for n in range(num_days):
        X_tmp, y_tmp = prepare_dataset(df_train_bo, tr_start + timedelta(days=n))
        X_li.append(X_tmp)
        y_li.append(y_tmp)
        gc.collect()

    X_tr_boi = pd.concat(X_li, axis=0)
    y_tr_boi = np.concatenate(y_li, axis=0)

    def lgbm_val(learning_rate, max_bin, num_leaves,
                 min_data_in_leaf,
                 # min_sum_hessian_in_leaf,
                 bagging_fraction,
                 #  bagging_freq,
                 feature_fraction,
                 lambda_l1, lambda_l2):

        params = {
            'boost_from_average': True,
            'boosting': 'gbdt',
            # 'Decay': 'linear',
            'n_estimators': 5000,
            'early_stopping_rounds': 200,
            'objective': 'regression_l2',
            'metric': 'l2',
            'num_threads': 16,
            'bagging_freq': 1,
            # 'max_depth':-1
        }

        # params['min_sum_hessian_in_leaf']= min_sum_hessian_in_leaf
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        # params['bagging_freq']= int(round(bagging_freq)) #k means perform bagging at every k iteration
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        params['learning_rate'] = learning_rate
        params['max_bin'] = int(round(max_bin))
        params['num_leaves'] = int(round(num_leaves))

        df_train_bo = df_train.copy()
        y_test_bo_rmse = []
        test_pred_bo = []
        X_l = X_li.copy()
        y_l = y_li.copy()
        X_tr_bo = X_tr_boi.copy()
        y_tr_bo = y_tr_boi.copy()

        for i in range(bo_forecast_period):
            X_v, y_v = [], []
            for m in range(val_period):
                X_tmp, y_tmp = prepare_dataset(df_train_bo, tr_start + timedelta(days=num_days + m + i))
                X_v.append(X_tmp)
                y_v.append(y_tmp)
                gc.collect()
            X_val_bo = pd.concat(X_v, axis=0)
            y_val_bo = np.concatenate(y_v, axis=0)

            X_test_bo, y_test_bo = prepare_dataset(df_train, bo_pred_start + timedelta(days=i))

            y_test_bo_rmse.append(y_test_bo)

            dtrain_bo = lgb.Dataset(X_tr_bo, label=y_tr_bo.reshape(-1, ), categorical_feature=cat_feat)
            dval_bo = lgb.Dataset(X_val_bo, label=y_val_bo.reshape(-1, ), categorical_feature=cat_feat,
                                  reference=dtrain_bo)
            bst = lgb.train(params, dtrain_bo, valid_sets=[dtrain_bo, dval_bo], verbose_eval=False)
            test_pred_bo.append(bst.predict(X_test_bo, num_iteration=bst.best_iteration))

            if i < (bo_forecast_period - 1):
                df_train_bo[df_train_bo.columns[-(forecast_period + bo_forecast_period - i)]] = bst.predict(X_test_bo,
                                                                                                            num_iteration=bst.best_iteration)
                X_tmp, y_tmp = prepare_dataset(df_train_bo, tr_start + timedelta(days=(i + num_days)))
                X_l.append(X_tmp)
                y_l.append(y_tmp)
                X_tr_bo = pd.concat(X_l, axis=0)
                y_tr_bo = np.concatenate(y_l, axis=0)
                gc.collect()

        return -((np.array(y_test_bo_rmse).reshape(bo_forecast_period, num_category) - np.array(
            test_pred_bo)) ** 2).sum()

    pds = {
        # 'min_sum_hessian_in_leaf':(0,1),
        'min_data_in_leaf': (5, 15),
        'feature_fraction': (0.2, 0.4),
        'bagging_fraction': [0.1, 0.3],
        # 'bagging_freq': [0,0.1],
        # 'lambda_l1': (0.001, 0.1),
        # 'lambda_l2': (0.001, 0.1),
        'learning_rate': (0.001, 0.1),
        'max_bin': (200, 700),
        # 'num_leaves': (10, 100)
    }

    optimizer = BayesianOptimization(lgbm_val, pds, random_state=random_state)
    # load_logs(optimizer,logs=['01log.json'])
    logger = JSONLogger('01log.json')
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max['params']


best_params = lgbm_bo(df_train)


# best_params['num_leaves'] = int(round(best_params['num_leaves']))
best_params['max_bin'] = int(round(best_params['max_bin']))
best_params['min_data_in_leaf'] = int(round(best_params['min_data_in_leaf']))
# best_params['bagging_freq'] = int(round(best_params['bagging_freq']))


logdata = [json.loads(line) for line in open('01log.json', 'r')]


target = []
param_log = pd.DataFrame(columns=list(logdata[0]['params'].keys()))
for iter_num in range(0, init_points + n_iter):
    param_log.loc[iter_num + 1] = list(logdata[iter_num]['params'].values())
    target.append(-logdata[iter_num]['target'])
param_log['target'] = target
param_log.sort_values('target', axis=0, ascending=True)


best_params = {**{
    'boost_from_average': True,
    'boosting': 'gbdt',
    # 'Decay': 'linear',
    'n_estimators': 5000,
    'early_stopping_rounds': 200,
    'objective': 'regression_l2',
    'metric': 'l2',
    'num_threads': 16,
    # 'max_depth': None,
    'bagging_freq': 1}, **best_params}


test_pred = []
feat_imp = []
tr_start = pred_start - timedelta(days=num_days + val_period)

X_l, y_l = [], []

for n in range(num_days):
    X_tmp, y_tmp = prepare_dataset(df_train, tr_start + timedelta(days=n))
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

for i in range(forecast_period):
    # print("=" * 50)
    # print("Step %d" % (i+1))
    # print("=" * 50)

    X_v, y_v = [], []

    for m in range(val_period):
        X_tmp, y_tmp = prepare_dataset(df_train, tr_start + timedelta(days=m + i + num_days))
        X_v.append(X_tmp)
        y_v.append(y_tmp)
        gc.collect()

    X_val = pd.concat(X_v, axis=0)
    y_val = np.concatenate(y_v, axis=0)

    X_test = prepare_dataset(df_train, pred_start + timedelta(days=i), is_train=False)

    dtrain = lgb.Dataset(X_train, label=y_train.reshape(-1, ), categorical_feature=cat_feat)

    dval = lgb.Dataset(X_val, label=y_val.reshape(-1, ), categorical_feature=cat_feat, reference=dtrain)

    bst = lgb.train(best_params, dtrain, valid_sets=[dtrain, dval], verbose_eval=False)

    # print("\n".join(("%s: %.2f" % x) for x in sorted(
    #       zip(X_train.columns, bst.feature_importance("gain")),
    #       key=lambda x: x[1], reverse=True
    #   )))
    feat_imp.append(bst.feature_importance("gain"))
    test_pred.append(bst.predict(X_test, num_iteration=bst.best_iteration))

    if i < (forecast_period - 1):
        df_train[df_train.columns[-(forecast_period - i)]] = bst.predict(X_test, num_iteration=bst.best_iteration)
        X_tmp, y_tmp = prepare_dataset(df_train, tr_start + timedelta(days=(i + num_days)))
        X_l.append(X_tmp)
        y_l.append(y_tmp)
        X_train = pd.concat(X_l, axis=0)
        y_train = np.concatenate(y_l, axis=0)
        gc.collect()


lgb.create_tree_digraph(bst, show_info=['split_gain', 'internal_value', 'leaf_count', 'internal_count'])


feat_eval = np.array(feat_imp).transpose()
df_feat_eval = pd.DataFrame(feat_eval, index = X_train.columns)
df_feat = df_feat_eval.copy()
df_feat['evg']=df_feat_eval.mean(axis=1)
df_feat['max']=df_feat_eval.max(axis=1)
df_feat['#0'] = (df_feat_eval==0).sum(axis=1).values
# df_feat['!0'] = (df_feat_eval>0).sum(axis=1).values

df_feat[['evg','max','#0']].sort_values(['evg'])
#df_feat.loc[X_train.columns.str.contains(pat='mean')][['evg','#0']]    # mean 같은 단어 기준 검색


y_val = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_val, index=df_train.index,
    columns=pd.date_range(te_start, periods=forecast_period)
).stack().to_frame("value")
df_preds.index.set_names(["ITEM", "WDATE"], inplace=True)
# df_preds["value"] = np.expm1(df_preds["value"])
# df_preds.reset_index().to_csv('lgb_0826.csv', index=False)


df_preds = df_preds.reset_index()


# df_preds = df_preds.pivot('WDATE', 'ITEM', 'value') * (sc.max() - sc.min()) + sc.min()
df_preds = df_preds.pivot('WDATE', 'ITEM', 'value')
# df_preds = np.expm1(df_preds)

a = a[a.WDATE >= te_start]


# a.value = np.expm1(a.value)


polo = a.pivot(index='ITEM', columns='WDATE', values='value').fillna(0)

croco = pd.DataFrame(columns=polo.columns[34:], index=df_preds.columns)

rmse = []
for item in df_preds.columns:
    #   rmse.append(mean_squared_error(a[a.ITEM==item]['value'],df_preds[df_preds.ITEM==item]['value'])**0.5)
    rmse.append(mean_squared_error(polo.loc[item], df_preds[item]) ** 0.5)


for item in df_preds.columns:
    residual = (np.array(df_preds[item]) - np.array(polo.loc[item])).tolist()
    #   residual = residual.to_list()
    cat = []
    for i in np.arange(35, len(residual) + 1):
        cat.append(sum(residual[i - 35:i]))
    croco.loc[item] = cat


croco['avg_35sum'] = croco.sum(axis=1) / len(croco.columns)
croco['max_35'] = croco.max(axis=1)
croco['min_35'] = croco.min(axis=1)
# croco['avg_35rev'] = a.groupby('ITEM').apply(sum)['value']*35/forecast_period
croco['avg_35rev'] = polo.sum(axis=1) * 35 / forecast_period

croco['avg_rate'] = croco['avg_35sum'] / croco['avg_35rev']
croco['max_rate'] = croco['max_35'] / croco['avg_35rev']
croco['min_rate'] = croco['min_35'] / croco['avg_35rev']
croco['RMSE'] = rmse


croco.drop(croco.columns[1:8].values, axis=1)


# from matplotlib import pyplot as plt
# for item in df_preds.columns:
#     plt.figure(figsize=(18, 5))
#     plt.plot(polo.columns, df_preds[item], label='prediction')
#     #   plt.plot(a[a.ITEM==item]['WDATE'],a[a.ITEM==item]['value'], label='actual')
#     plt.plot(polo.columns, polo.loc[item], label='actual')
#     plt.legend(loc='upper left')
# #   plt.xticks(a[a.ITEM==item]['WDATE'].values[np.arange(0,forecast_period,7)].tolist(),rotation=60)
#
#
