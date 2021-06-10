from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np

import warnings
warnings.filterwarnings('ignore')

class Stacking():
    
    def __init__(self, models, meta_alg, *, cv=5, metrics=[roc_auc_score], metrics_params={}, meta_weight=1, base_weight=1):
        self.models = models
        self.meta_alg = meta_alg
        self.cv = cv
        self.metrics = metrics
        self.metrics_params = metrics_params
        self.base_predicts_ = None
        self.meta_weight = meta_weight
        if base_weight == 1:
            self.base_weight = [1 for i in range(len(models))]
        else:
            self.base_weight = base_weight
    
    def fit(self, X, y):
        self.fit_base(X, y)
        self.fit_meta(X, y)
    
    def fit_base(self, X, y):

        # fitting base algorithms and getting prediction matrix of all base models for train dataset
        for base_algotithm in self.models:
            base_algotithm.fit(X, y)
        
    def fit_meta(self, X, y):
        # build an empty matrix of meta features for train dataset
        feat_mtrx = np.empty((X.shape[0], len(self.models)))
        
        for n, base_algotithm in enumerate(self.models):
            feat_mtrx[:, n] = cross_val_predict(base_algotithm, X, y, cv=self.cv, method='predict')
        
        # fitting meta algorithm using pred matrix of base models as meta features
        self.meta_alg.fit(feat_mtrx, y)
        
    def predict(self, X):
        # meta predictions for test dataset as actual values
        meta_pred = self.predict_proba(X) * self.meta_weight
        meta_pred = np.argmax(meta_pred, axis=1)
        
        return meta_pred
        
    def predict_proba(self, X):
        self.base_predicts_ = self.predict_base(X)
        
        # meta predictions for test dataset probability values
        meta_proba = self.meta_alg.predict_proba(self.base_predicts_)
        
        return meta_proba
        
    def predict_base(self, X, model=None):
        
        if model == None:
            model_list = self.models
        else:
            model_list = [model]
        
        # build an empty matrix of meta features for test dataset
        feat_mtrx = np.empty([X.shape[0], len(model_list)])
        
        # getting predictions of base algorithms for test dataset
        for n, model in enumerate(model_list):
            pred = self.predict_base_proba(X, model) * self.base_weight[n]
            pred = np.argmax(pred, axis=1)
            feat_mtrx[:, n] = pred
            
        return feat_mtrx
    
    def predict_base_proba(self, X, model):  
        return model.predict_proba(X)
    
    def score(self, X, y_true, print_scores=True):
        scores = {}
        for metric in self.metrics:
            # get params as **kwargs for current metric
            if type(self.metrics_params.get(metric.__name__)) is dict:
                params = self.metrics_params.get(metric.__name__)
            else:
                params = {}
            # print output of each metric
            try:
                scores.update({metric.__name__: metric(y_true, self.predict(X), **params)})
            # AxisError, few metrics working with probabilities and not with actual values
            except Exception:
                scores.update({metric.__name__, metric(y_true, self.predict_proba(X), **params)})
                
        if print_scores == True:
            self.print_scores(scores)
        return scores
                
    def score_base(self, X, y_true, model_labels, print_scores=True):
        
        labeled_scores = {}
        for model_label, model in zip(model_labels, self.models):
            scores = {}
            for metric in self.metrics:
                # get params as **kwargs for current metric
                if type(self.metrics_params.get(metric.__name__)) is dict:
                    params = self.metrics_params.get(metric.__name__)
                else:
                    params = {}
                # print output of each metric
                try:
                    scores.update({metric.__name__: metric(y_true, self.predict_base(X, model), **params)})
                except:
                    scores.update({metric.__name__: metric(y_true, self.predict_base_proba(X, model), **params)})
            labeled_scores.update({model_label: scores})
        
        if print_scores == True:
            for model_label, scores in labeled_scores.items():
                star = '****************************************************'
                star_len = len(star)
                gap = ''.join([' ' for i in np.arange(star_len / 2 - np.ceil(len(model_label) / 2))])
                print(f'\n{star}\n{gap}{model_label}\n{star}')
                self.print_scores(scores)
    
        return labeled_scores
        
    def print_scores(self, scores):
        for metric_label, score in scores.items():
            print('--------------------------')
            print(f'{metric_label}:\n{score}')
        print('--------------------------')    
