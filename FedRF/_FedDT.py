""" A local implementation of a federated decision tree classifier based on a secure multi-parties float-point computation protocol """

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


from ._base_test import BaseFederatedDecisionTree, base_splitter, base_Tree, base_NodeInf


from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args

from FedHC import glb 

class splitter(base_splitter):
    def __init__(self, n_features, n_splits):
        self.n_features = n_features
        self.n_splits = n_splits
        super().__init__([]) 
        self.split_index = np.ones((n_features, n_splits), dtype=bool)
        self._is_bulit = False
      


class Tree(base_Tree):
    def __init__(self):
        #self.feature_id = [-1 for _ in range(n_features)]
        # Use node_id as entry to record the information and data_id of each node
        self.node_inf = {}
        self.data_id = {}
        self.feature_bound = {}
        #self.end_node = -1
        self.leaf_node = []

class NodeInf(base_NodeInf):
    def __init__(self, node_id = None, gini = None, best_feature_id=None, best_split=None, min_gini_split=None, majority_class=None):
        self.node_id = node_id
        self.gini = gini
        self.best_feature_id = best_feature_id
        self.best_split = best_split
        self.min_gini_split = min_gini_split
        self.majority_class = majority_class
        #self.left = None
        #self.right = None
        #self.data_id = data_id

class FederatedDecisionTreeClassifier(MultiOutputMixin, BaseFederatedDecisionTree, ClassifierMixin):
    
    @_deprecate_positional_args
    def __init__(self, *,
                  max_features=None,
                  random_state=None,
                  n_jobs=None,
                  n_splits=20,
                  verbose=0):
        #self.min_samples_split = min_samples_split
        #self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_splits = n_splits
        self.verbose = verbose
        self.n_jobs = n_jobs

        self._is_fitted = False
    
    def _check_is_fitted(self):
        if self._is_fitted:
            return None
        else:
            raise NotFittedError("This %s instance is not fitted yet. Call 'fit' before using this estimator." 
                                 % type(self).__name__)
    
    
    def fit(self, X, y):
        """ A method to fit the Federated decision tree to federated X and y
        
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

        glb._init()
        glb.set_value('communication', 0)
        
        # TODO: Other values of max_features_
        self.n_features_ = X[0].shape[1]
        if self.max_features == 'auto':
            self.max_features_ = np.int(np.sqrt(self.n_features_))
        elif self.max_features == None:
            self.max_features_ = self.n_features_
        else:
            raise ValueError("max_features can only be 'auto' or None")
        
        self._random_state = check_random_state(self.random_state)
        
        #each party has all class labels
        #self._distinct_class_vals = class_distribution(np.asarray(y[num_parties]).reshape(-1, 1))[0][0]
        
        # no need each party to have all class labels
        self._distinct_class_vals = np.unique(np.concatenate([y[party_id] for party_id in range(num_parties)]))
        
        # Each party publish the upper and lower bound of each feature of the local data
        # The initiator calculates the global upper bound and lower bound
        # then selects self.n_split split candidates randomly from [feature_lower, feature_upper)
        
        # self._splitter = splitter(self.n_features_, self.n_splits)
        feature_bound = []
        for i in range(self.n_features_):
            feature_upper = max([np.max(X[party_id][:, i])\
                                   for party_id in range(num_parties)])
            feature_lower = min([np.min(X[party_id][:, i])\
                                   for party_id in range(num_parties)])
            feature_bound.append([feature_lower, feature_upper])
        
        #self._splitter.build(feature_bound, self._random_state)
        
        # Build tree
        self._tree = Tree()
        
        # initialize the root node
        root_id = 1
        self._tree.data_id[root_id] = [np.arange(len(X[i])) for i in range(num_parties)]
        self._tree.node_inf[root_id] = NodeInf(root_id)
        self._tree.feature_bound[root_id] = feature_bound

        # stack for depth-first tree builder 
        stack = []
        stack.append(root_id)
        
        # Stop when no node can be split
        while(len(stack) > 0):
            internal_node = stack.pop()
            
            if self._check_node(internal_node, X, y):
                #feature_set.remove(self._tree.node_inf[internal_node].best_feature_id)
                stack.append(internal_node * 2 + 1)
                stack.append(internal_node * 2)
                
        
        self._is_fitted = True

        self.communication_amount = glb.get_value('communication')

        return self

    def _find_split(self, node_id, X, y):    
        num_parties = len(X)
        data_id_this = self._tree.data_id[node_id]
        visiting_features = []
        if self.max_features_ < self.n_features_:
            visiting_features = self._random_state.choice(
                                  list(range(self.n_features_)), self.max_features_, replace=False)
        else:
            visiting_features = np.arange(self.n_features_)


        F_stat_of_allfeatures = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                        **_joblib_parallel_args(backend='loky'))(
            delayed(self._parallel_check_feature)(
                node_id, current_feature, X, y)
                for current_feature in visiting_features)
        
        parallel_com = [result[1] for result in F_stat_of_allfeatures]
        F_stat_of_allfeatures = [result[0] for result in F_stat_of_allfeatures]

        glb.set_value('communication', glb.get_value('communication') + sum(parallel_com))

        visiting_features = visiting_features[\
                np.argsort(F_stat_of_allfeatures)[::-1][:np.int32(np.log2(self.max_features_) + 1)]]

        min_gini = np.inf
        best_feature = None
        best_split = None
        data_id_left_split = None
        data_id_right_split = None
        gini_left_split = None
        gini_right_split = None
        
        # find the best split

        # version 3: get all feature-split pairs and check each pair concurrently
        # -----------------------------------------------------

        # feature_split_candidates = []
        # for feature in visiting_features:
        #     split_candidates = self._splitter.get_split(feature)
        #     feature_split_candidates.extend(list(zip([feature] * len(split_candidates), split_candidates)))

        # all_gini = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
        #                                 **_joblib_parallel_args(backend='loky'))(
        #     delayed(self._parallel_check_feature_split)(
        #         node_id, feature_split, X, y)
        #         for feature_split in feature_split_candidates)
        
        
        
        # # note: np.argmin takes all_gini as a 2-D array and does element-wise comparision rather than comparing each triple,
        # #       thus we manully select the first element of each triple, i.e., the gini of each feature-split pair for np.argmin.
        # #       If we just use np.argmin(all_gini), the return is the index of the minimum of all the elements, which is not expected.
        # min_gini_idx = np.argmin( np.array(all_gini)[:,0] ) 
        # if min_gini_idx >= len(feature_split_candidates) or min_gini_idx < 0:
        #     print(all_gini)
        #     raise ValueError("index: %d, len(all_gini)=%d, len(fea..)=%d" % (min_gini_idx, len(all_gini), len(feature_split_candidates)))

        # min_gini, gini_left_split, gini_right_split = all_gini[min_gini_idx]
        # best_feature, best_split = feature_split_candidates[min_gini_idx]

        # ----------------------------------------------------------


        # version 2, check each feature concurrently
        # --------------------------------------------------------------

        best_split_of_allfeatures = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                        **_joblib_parallel_args(backend='loky'))(
            delayed(self._parallel_check_split)(
                node_id, current_feature, X, y)
                for current_feature in visiting_features)

        parallel_com = [result[1] for result in best_split_of_allfeatures]
        best_split_of_allfeatures = [result[0] for result in best_split_of_allfeatures]
        
        glb.set_value('communication', glb.get_value('communication') + sum(parallel_com))

        min_gini_idx = np.argmin( np.array(best_split_of_allfeatures)[:,0] ) 
        min_gini, best_feature, best_split, gini_left_split, gini_right_split = best_split_of_allfeatures[min_gini_idx]

        # for result in best_split_of_allfeatures:
        #     if result[0] < min_gini:
        #         min_gini, best_feature, best_split, gini_left_split, gini_right_split = result
        
        # -----------------------------------------------------------------


        # version 1, sequentially check each split    
        # -----------------------------------------------------------------

        # best_split_of_allfeatures = []
        # for current_feature in visiting_features:
        #     best_split_of_allfeatures.append(self._parallel_check_split(node_id, current_feature, X, y))
        # for result in best_split_of_allfeatures:
        #     if result[0] < min_gini:
        #         min_gini, best_feature, best_split, gini_left_split, gini_right_split = result
        
        # -------------------------------------------------------------------

        
    


        # for feature in visiting_features:
#             feature_upper = max([np.max(X[party_id][self._tree.data_id[node_id][party_id]][:, feature])\
#                                    for party_id in range(num_parties)])
#             feature_lower= min([np.max(X[party_id][self._tree.data_id[node_id][party_id]][:, feature])\
#                                    for party_id in range(num_parties)])
            # split_candidates = self._splitter.get_split(feature)
            # for split in split_candidates:
            #     data_id_left = [data_id_this[party_id][
            #                     np.where(X[party_id][data_id_this[party_id]][:, feature] <= split[1])[0]]\
            #                          for party_id in range(num_parties)]
            #     data_id_right = [data_id_this[party_id][
            #                     np.where(X[party_id][data_id_this[party_id]][:, feature] > split[1])[0]]\
            #                          for party_id in range(num_parties)]
                
            #     num_ins_array_left, num_ins_by_class_left = self._count_instances(data_id_left, y)
            #     num_ins_array_right, num_ins_by_class_right = self._count_instances(data_id_right, y)
                
            #     if num_ins_array_left is not None and num_ins_array_right is not None:
            #         gini_left = _secure_gini(num_ins_array_left, num_ins_by_class_left)
            #         gini_right = _secure_gini(num_ins_array_right, num_ins_by_class_right)
            #         if gini_left + gini_right < min_gini:
            #             min_gini = gini_left + gini_right
            #             best_feature = feature
            #             best_split = split
            #             data_id_left_split = data_id_left
            #             data_id_right_split = data_id_right
            #             gini_left_split = gini_left
            #             gini_right_split = gini_right
        
        # if all existing splits cannot split this node, then ignore splitting and return False
        if min_gini == np.inf or best_split is None:
            return False

        
        data_id_left_split = [data_id_this[party_id][
                        np.where(X[party_id][data_id_this[party_id]][:, best_feature] <= best_split)[0]]\
                             for party_id in range(num_parties)]
        data_id_right_split = [data_id_this[party_id][
                        np.where(X[party_id][data_id_this[party_id]][:, best_feature] > best_split)[0]]\
                             for party_id in range(num_parties)]
            
        self._tree.node_inf[node_id].set_split_info(best_feature, best_split, min_gini)
        
        left_child_id = node_id * 2
        self._tree.data_id[left_child_id] = data_id_left_split
        self._tree.node_inf[left_child_id] = NodeInf(left_child_id, gini_left_split)
        
        right_child_id = left_child_id + 1
        self._tree.data_id[right_child_id] = data_id_right_split
        self._tree.node_inf[right_child_id] = NodeInf(right_child_id, gini_right_split)
        
        #self._splitter.take_split(best_feature, best_split)
        self._tree.feature_bound[left_child_id] = self._tree.feature_bound[node_id]
        self._tree.feature_bound[left_child_id][best_feature][1] = best_split
        self._tree.feature_bound[right_child_id] = self._tree.feature_bound[node_id]
        self._tree.feature_bound[right_child_id][best_feature][0] = best_split

        if self.verbose > 1:
            print("split node id: {0:d}\n"
                  "feature id: {1:d}\n"
                  "split: ({2:f})\n"
                  "gini: {3:f}\n".format(node_id, best_feature, best_split, min_gini))
        # print("feature rank: %d" % np.where(visiting_features == best_feature)[0])

        return True

    def apply(self, X):
        self._check_is_fitted()
        return self._tree.apply(X)
        
        
    
    
    def predict(self, X):
        """Return the predicted label of each sample in X
        
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
        y_pred_idx = np.zeros(len(X), dtype=int)
        y_pred = np.zeros(len(X), dtype=self._distinct_class_vals.dtype)
        X_leafs = self._tree.apply(X)
        for i in range(len(X_leafs)):
            y_pred_idx[i], y_pred[i] = self._tree.node_inf[X_leafs[i]].majority_class
        
        return y_pred_idx, y_pred


    
 

