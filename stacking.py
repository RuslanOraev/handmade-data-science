from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np

import warnings
warnings.filterwarnings('ignore')


def stacking(models, meta_alg, data_train, targets_train, data_test, targets_test=None, random_state=None, test_size=None, cv=5, metrics=[roc_auc_score], metrics_params={}):
    
    '''
    |
    Stacking and blending algorithms.
    
    Parameters
    ----------
    models: list of ML models using method `fit`
    
    meta_alg: classification meta-algorithm for blending or stacking using method `fit`
    
    data_train: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    
    targets_train: array-like of shape (n_samples,)
        Target vector relative to X.
        
    data_test: {array-like, sparse matrix} of shape (n_samples, n_features)
        Validation vector, where n_samples is the number of samples and
        n_features is the number of features.
    
    targets_test: array-like of shape (n_samples,), default=None
        Validation target vector relative to X.
    
    random_state: int, RandomState instance, default=None
        renewable randomizer that using in blending for data split
    
    test_size: float or None, default=None
        Size of validation data for blending.
        If float is ``0 < value < 1``, than `blending` is ON
        If None, than `stacking` is ON
        
    cv: int or None
        Numb of folds for cross validation
    
    metrics: list of objects, default=[roc_auc_score]
        Metrics using construction `metric(y_true, y_pred)`
    
    metrics_params: dict of dicts, default={}
        `{'metric': {'param_1':'value_1', 'param_2':'value_2'},
        'metric_2': {'param_2_1':'value_2_1', 'param_2_2':'value_2_2'}}`
    
    Examples
    -------
    
In[1]:
>>> from sklearn.datasets import load_wine
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.svm import SVC
>>> from xgboost import XGBClassifier

>>> x_train, x_test, y_train, y_test = train_test_split(load_wine().data, 
...                                                     load_wine().target)

>>> knn = KNeighborsClassifier(n_neighbors=3)
>>> lr = LogisticRegression(random_state=17)
>>> svc = SVC(random_state=17)

>>> meta = XGBClassifier(n_estimators=40, eval_metric='logloss')
>>> models = [knn, lr, svc]

>>> from sklearn.metrics import accuracy_score, confusion_matrix, \\
...     precision_score, recall_score, f1_score, roc_auc_score, \\
...     precision_recall_fscore_support, classification_report

>>> metrics = [roc_auc_score, accuracy_score, confusion_matrix, 
...     precision_score, recall_score, f1_score, 
...     precision_recall_fscore_support, classification_report]

>>> metrics_params = {'roc_auc_score': {'multi_class':'ovo', 'average':'macro'},
...                   'precision_score': {'average':'macro'},
...                   'recall_score': {'average':'macro'},
...                   'f1_score': {'average':'macro'}
...                  }

>>> stacking(models, 
...          meta_alg=meta,
...          data_train=x_train,
...          targets_train=y_train,
...          data_test=x_test,
...          targets_test=y_test,
...          random_state=None,
...          test_size=None,
...          cv=5,
...          metrics=metrics,
...          metrics_params=metrics_params
...         )

Out[1]:
roc_auc_score:
0.9706349206349207

accuracy_score:
0.9555555555555556

confusion_matrix:
[[13  1  0]
 [ 0 15  1]
 [ 0  0 15]]

precision_score:
0.9583333333333334

recall_score:
0.9553571428571429

f1_score:
0.9560682994822779

precision_recall_fscore_support:
(array([1.    , 0.9375, 0.9375]), array([0.92857143, 0.9375    , 1.        ]), array([0.96296296, 0.9375    , 0.96774194]), array([14, 16, 15], dtype=int64))

classification_report:
              precision    recall  f1-score   support

           0       1.00      0.93      0.96        14
           1       0.94      0.94      0.94        16
           2       0.94      1.00      0.97        15

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.96        45
weighted avg       0.96      0.96      0.96        45
    '''
    
    # classical stacking
    if test_size is None:
        # build an empty matrix of meta features for train dataset
        meta_mtrx = np.empty((data_train.shape[0], len(models)))
        
        # fitting base algorithms and getting prediction matrix of all base models for train dataset
        for n, base_algotithm in enumerate(models):
            meta_mtrx[:, n] = cross_val_predict(base_algotithm, data_train, targets_train, cv=cv, method='predict')
            base_algotithm.fit(data_train, targets_train)
            
        # fitting meta algorithm using pred matrix of base models as meta features
        meta_model = meta_alg.fit(meta_mtrx, targets_train)
        
        # build an empty matrix of meta features for test dataset
        meta_mtrx_test = np.empty([data_test.shape[0], len(models)])
        
        # getting predictions of base algorithms for test dataset
        for n, base_algotithm in enumerate(models):
            meta_mtrx_test[:, n] = base_algotithm.predict(data_test)
        
        # meta predictions for test dataset as actual and probability values
        meta_pred = meta_model.predict(meta_mtrx_test)
        meta_proba = meta_model.predict_proba(meta_mtrx_test)
        
        # print metrics if there's some validation data
        if targets_test is not None:
            for metric in (metrics):
                # get params as **kwargs for current metric
                if type(metrics_params.get(metric.__name__)) is dict:
                    params = metrics_params.get(metric.__name__)
                else:
                    params = {}
                # print output of each metric
                try:
                    print(f'{metric.__name__}:\n{metric(targets_test, meta_pred, **params)}\n')
                # AxisError, few metrics working with probabilities and not with actual values
                except Exception:
                    print(f'{metric.__name__}:\n{metric(targets_test, meta_proba, **params)}\n')
        else:
            print('Не поступило данных для проверки')
            return(meta_pred)
        
            
    # blending
    elif test_size > 0 and test_size < 1:
        
        # split data to train and valid
        train, valid, train_true, valid_true = train_test_split(data_train,
                                                                targets_train,
                                                                test_size=test_size,
                                                                random_state=random_state)
        
        # build an empty matrix of meta features for train dataset
        meta_mtrx = np.empty((valid.shape[0], len(models)))
        
        # fitting base algorithms and getting prediction matrix of all base models for train dataset
        for n, model in enumerate(models):
            model.fit(train, train_true)
            meta_mtrx[:, n] = model.predict(valid)
            
        # fitting meta algorithm using pred matrix of base models as meta features
        meta_model = meta_alg.fit(meta_mtrx, valid_true)
        
        # build an empty matrix of meta features for test dataset
        meta_mtrx_test = np.empty((data_test.shape[0], len(models)))
        
        # getting predictions of base algorithms for test dataset
        for n, model in enumerate(models):
            meta_mtrx_test[:, n] = model.predict(data_test)
        
        # meta predictions for test dataset as actual and probability values
        meta_pred = meta_model.predict(meta_mtrx_test)
        meta_proba = meta_model.predict_proba(meta_mtrx_test)
        
        # print metrics if there's some validation data
        if targets_test is not None:
            for metric in (metrics):
                # get params as **kwargs for current metric
                if type(metrics_params.get(metric.__name__)) is dict:
                    params = metrics_params.get(metric.__name__)
                else:
                    params = {}
                # print output of each metric
                try:
                    print(f'{metric.__name__}:\n{metric(targets_test, meta_pred, **params)}\n')
                # AxisError, few metrics working with probabilities and not with actual values
                except Exception:
                    print(f'{metric.__name__}:\n{metric(targets_test, meta_proba, **params)}\n')
        else:
            print('Не поступило данных для проверки')
            return(meta_pred)
    
    else:
        raise ValueError("test_size must be between 0 and 1")