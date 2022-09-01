""" A local implementation of a federated random forest classifier based on a secure multi-parties float-point computation protocol """

__author__ = "Zhiyu Liang"


# import sys

# sys.path.append('../')

import threading

# from numba import jit

import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator
from sklearn.base import MultiOutputMixin
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state
from sklearn.utils import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import class_distribution
#from sklearn.utils.validation import check_is_fitted

from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args



from ._FedDT import FederatedDecisionTreeClassifier



def _parallel_build_trees(tree, X, y, tree_idx, n_trees, verbose=0):
    """ Private function used to fit a single tree in parallel.
    
    """
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))
    
    tree.fit(X, y)
    
    return tree


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally, because joblib
    complains that it cannot pickle it when placed there.
    """

    prediction_idx, _ = predict(X)
    with lock:
        for i in range(len(prediction_idx)):
            out[i][prediction_idx[i]] += 1


    
class FederatedRandomForestClassifier(MultiOutputMixin, BaseEnsemble, ClassifierMixin):
    
    
    @_deprecate_positional_args
    def __init__(self, 
                  n_estimators=100, *,
                  max_features='auto',
                  n_jobs=None,
                  random_state=None,
                  n_splits=20,
                  verbose=0):
        super().__init__(
            base_estimator=FederatedDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("max_features", "random_state", "n_splits", "n_jobs", "verbose"))
            
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_splits = n_splits
        self.verbose = verbose
        
        self._is_fitted = False
   
    def _check_is_fitted(self):
        if self._is_fitted:
            return None
        else:
            raise NotFittedError("This %s instance is not fitted yet. Call 'fit' before using this estimator." 
                                 % type(self).__name__)
            
    def fit(self, X, y):
        """ A method to fit the Federated random forest classifier to federated X and y
        
            Parameters
            -------------------------
            X: list of array-like of shape (n_samples, n_features).
                The federated training data
                
            y: list of array-like of shape (n_samples,) or (n_samples, 1)
                The class values of X
                
            
            Return
            --------------------------
            self: This estimator
        """
        num_parties = len(X)
        
        self._random_state = check_random_state(self.random_state)

        
        
        # no need each party to have all class labels
        self._distinct_class_vals = np.unique(np.concatenate([y[party_id] for party_id in range(num_parties)]))
        
        self._validate_estimator()
        trees = [self._make_estimator(append=False,
                                      random_state=self._random_state)
                for i in range(self.n_estimators)]
        
        self.estimators_ = []
        
        trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_trees)(
                t, X, y, i, len(trees), self.verbose)
            for i, t in enumerate(trees))
        
        self.estimators_.extend(trees)
        
        self.communication_amount = sum([clf.communication_amount for clf in self.estimators_])

        self._is_fitted = True
        
        return self
    
    def predict(self, X):
        """Return the predicted label of each sample in X.
            The predicted label corresponds to the class value yielding to the maximal probability.
        
        Parameters:
        -----------------------------
        X：Array-like of shape (n_samples, n_features)
           Unlike the fit method where the input is federated, here X is located in the initiator. The initiator uses the
           learned tree for inference. This corresponds to the setting of our framework. 
           
        Returns:
        -----------------------------
        y_pred: Array-like of shape(n_samples,)
            The predicted class values of associated samples in X
        
        """
        self._check_is_fitted()
        
        all_proba = self.predict_proba(X)

        # if the maximal predicted probability appears in more than one class, randomly choose one of them as the label.
        y_pred_idx = np.empty(all_proba.shape[0], dtype=int)
        for i in range(len(y_pred_idx)):
            y_pred_idx[i] = self._random_state.choice(np.where(all_proba[i] == np.max(all_proba[i]))[0]) 

        # return self._distinct_class_vals[np.argmax(all_proba, axis=1)]
        return self._distinct_class_vals[y_pred_idx]
    
    def predict_proba(self, X):
        """Return the predicted probability of each sample in X.
            The probability of a class is the fraction of the votes this class has gotten.
            The predicted class label of each tree is a vote to the corresponding class.
        
        Parameters:
        -----------------------------
        X：Array-like of shape (n_samples, n_features)
           Unlike the fit method where the input is federated, here X is located in the initiator. The initiator uses the
           learned tree for inference. This corresponds to the setting of our framework. 
           
        Returns:
        -----------------------------
        all_proba: Array-like of shape(n_samples, n_classes)
            The predicted class probability of associated samples in X
        
        """
            
        self._check_is_fitted()
        
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        
        all_proba = np.zeros((X.shape[0], self._distinct_class_vals.shape[0]))
        
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict, X, all_proba, lock)
            for e in self.estimators_)
        
        all_proba /= len(self.estimators_)
        
        return all_proba
        