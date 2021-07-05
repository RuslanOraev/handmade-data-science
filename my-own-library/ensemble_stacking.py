# import libraries
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Stacking():
    
    def __init__(self,  models, meta_alg=None, *, cv=5, metrics=[roc_auc_score], metrics_params={}, meta_weight=1, base_weight=1):
        # initial params
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
    
    def get_params(self, deep=True):
        return self.meta_alg.get_params(deep)
    
    def fit(self, X, y, base_fit_params={}, meta_fit_params={}):
        # fit base and meta models
        self.fit_base(X, y, base_fit_params)
        self.fit_meta(X, y, **meta_fit_params)
    
    def fit_base(self, X, y, fit_params={}):
        # fit base algorithms
        for base_algotithm in self.models:
            if base_algotithm in fit_params.keys():
                base_algotithm.fit(X, y, **fit_params[base_algotithm])
            else:
                base_algotithm.fit(X, y)
        
    def fit_meta(self, X, y, **kwargs):
        # build an empty matrix of meta features (base models predicts)
        feat_mtrx = np.empty((X.shape[0], len(self.models)))
        
        # getting prediction matrix of all base models using cross val
        for n, base_algotithm in enumerate(self.models):
            feat_mtrx[:, n] = cross_val_predict(base_algotithm, X, y, cv=self.cv, method='predict')
        
        # fitting meta algorithm using pred matrix of the base models as meta features
        self.meta_alg.fit(feat_mtrx, y, **kwargs)
        
    def predict(self, X):
        # meta predictions as probability values
        meta_pred = self.predict_proba(X) 
        # meta predictions as actual values
        meta_pred = np.argmax(meta_pred, axis=1)
        return meta_pred
        
    def predict_proba(self, X):
        # base predicts as actual values
        self.base_predicts_ = self.predict_base(X)
        # meta predictions as probability values
        meta_proba = self.meta_alg.predict_proba(self.base_predicts_) * self.meta_weight
        return meta_proba
        
    def predict_base(self, X, model=None, weight=1):
        
        # building weight list
        if model == None:
            model_list = self.models
        else:
            model_list = [model]
            
        if weight == 1:
            weight = self.base_weight
        else:
            weight = [weight]
        
        # build an empty matrix of meta features (base models predicts)
        feat_mtrx = np.empty([X.shape[0], len(model_list)])
        
        # getting predictions of fitted base algorithms 
        for n, model in enumerate(model_list):
            pred = self.predict_base_proba(X, model, weight[n])
            pred = np.argmax(pred, axis=1)
            feat_mtrx[:, n] = pred
        return feat_mtrx
    
    def predict_base_proba(self, X, model, weight=1): 
        # weighted predict_proba
        return model.predict_proba(X) * weight
    
    def score(self, X, y_true, print_scores=True):
        # dict with scrores
        scores = {}
        
        for metric in self.metrics:
            # get params for current metric
            if type(self.metrics_params.get(metric.__name__)) is dict:
                params = self.metrics_params.get(metric.__name__)
            else:
                params = {}
            
            # special algorithm for roc_auc
            if metric.__name__ == 'roc_auc_score':
                n_classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=np.append(n_classes, n_classes.max()))
                
                # if there more than 2 classes, roc auc calculated for each class
                if len(n_classes) > 2: 
                    searching_list = n_classes
                    class_desc = lambda n: f'{n} class '
                else: 
                    searching_list = [1]
                    class_desc = lambda n: ''
                for n in searching_list:
                    scores.update({f'{class_desc(n)}{metric.__name__}': metric(y_true_bin[:, n], self.predict_proba(X)[:, n], **params)})
                   
            else:
                # predict with actual values and calc the metric
                try:
                    scores.update({metric.__name__: metric(y_true, self.predict(X), **params)})
                # AxisError, few metrics working with probabilities and not with actual values
                except Exception:
                    scores.update({metric.__name__: metric(y_true, self.predict_proba(X), **params)})
                    
        # print score of each metric        
        if print_scores == True:
            self.print_scores(scores)
        return scores
                
    def score_base(self, X, y_true, model_labels, print_scores=True):
        
        # dict with model name scores
        model_scores = {}
        
        for n, (model_label, model) in enumerate(zip(model_labels, self.models)):
            # dict with scores
            scores = {}
            for metric in self.metrics:
                # get params for current metric
                if type(self.metrics_params.get(metric.__name__)) is dict:
                    params = self.metrics_params.get(metric.__name__)
                else:
                    params = {}
                    
                # special algorithm for roc_auc
                if metric.__name__ == 'roc_auc_score':
                    n_classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=np.append(n_classes, n_classes.max()))
                    
                    # if there more than 2 classes, roc auc calculated for each class
                    if len(n_classes) > 2: 
                        searching_list = n_classes
                        class_desc = lambda n: f'{n} class '
                    else: 
                        searching_list = [1]
                        class_desc = lambda n: ''
                    for n in searching_list:
                        scores.update({f'{class_desc(n)}{metric.__name__}': metric(y_true_bin[:, n], self.predict_base_proba(X, model, self.base_weight[n])[:, n], **params)})
                else:
                    # predict with actual values and calc the metric
                    try:
                        scores.update({metric.__name__: metric(y_true, self.predict_base(X, model, self.base_weight[n]), **params)})
                    # AxisError, few metrics working with probabilities and not with actual values
                    except:
                        scores.update({metric.__name__: metric(y_true, self.predict_base_proba(X, model, self.base_weight[n]), **params)})
                        
            # update model's scores
            model_scores.update({model_label: scores})
            
        # print scores of each metric of each model
        if print_scores == True:
            for model_label, scores in model_scores.items():
                star = '****************************************************'
                star_len = len(star)
                gap = ''.join([' ' for i in np.arange(star_len / 2 - np.ceil(len(model_label) / 2))])
                print(f'\n{star}\n{gap}{model_label}\n{star}')
                self.print_scores(scores)
    
        return model_scores
  
    def print_scores(self, scores):
        # print formatted scores
        for metric_label, score in scores.items():
            print('--------------------------')
            print(f'{metric_label}:\n{score}')
        print('--------------------------')    
