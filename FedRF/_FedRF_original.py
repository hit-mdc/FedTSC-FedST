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

from collections import Counter

from ..secret_sharing_based_computation._sfc import fl
from ..secret_sharing_based_computation._basic_building_blocks import basic_building_blocks as bbb


# secure gini computation based on sfc
# result is |D|Gini(D)

def _secure_gini(num_ins_array, num_ins_by_class_array):
    num_parties = num_ins_array.shape[1]
    FL = fl(9, 23, num_parties)
    #basci_operator = bbb(num_parties)
    # num_ins_array and num_ins_by_class_array are in the format or shared-integer, thus can be converted
    # into shared float-point format directly
            
    num_ins_shared = FL.Int2FL(num_ins_array, 32, 23)
    # repeat num_ins to the same shape with num_ins_by_class for the sake of array-based sfc
    num_ins_shared_expanded = FL.Int2FL(np.repeat(num_ins_array, len(num_ins_by_class_array),axis=0), 32, 23)
    #print(FL.fl_to_real(num_ins_shared))
    num_ins_by_class_shared = FL.Int2FL(num_ins_by_class_array, 32, 23)
        
    gini, error = FL.FLDiv(num_ins_by_class_shared, num_ins_shared_expanded)
            
    gini = FL.FLSub(num_ins_shared, FL.FLMul(FL.FLSum(FL.FLMul(gini, gini)), num_ins_shared))
    
    #gini = FL.FLSum(FL.FLMul(gini, gini))
    #gini = FL.FLMul(num_ins_shared, FL.FLSub(FL.real_to_fl(np.array([1])), gini))       
    
    return FL.fl_to_real(gini)[0,0]


# secure find the maxmum of a set of shared integers

def _find_majority(x):
    """ x.shape = (num_integers, num_parties) where num_integers >= 1 

    """
    if x.ndim != 2:
        raise ValueError("The dimension of should be 2 but %d is given." % x.ndim)
    if x.shape[0] == 1:
        return x
    num_parties = x.shape[1]
    basci_operator = bbb(num_parties)
    majority_index = 0
    for i in range(1, x.shape[0]):
        if basci_operator.Output(basci_operator.LT(x[majority_index].reshape(1,-1), x[i].reshape(1,-1), 32))[0,0]:
            majority_index = i
    
    return majority_index

# secure claculate the probability of each class in a node
def _cal_class_prob(x):
    """ 
    
    Parameters:
    -------------------------------
    num_ins_by_class    shape = (num_classes, num_parties) where num_classes >= 1 

    Returns:
    class_prob          shape = (num_classes, )
    
    """


class FederatedDecisionTreeClassifier(MultiOutputMixin, BaseEstimator, ClassifierMixin):
    
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
        self._splitter = splitter(self.n_features_, self.n_splits)
        feature_bound = []
        for i in range(self.n_features_):
            feature_upper = max([np.max(X[party_id][:, i])\
                                   for party_id in range(num_parties)])
            feature_lower = min([np.min(X[party_id][:, i])\
                                   for party_id in range(num_parties)])
            feature_bound.append([feature_lower, feature_upper])
        
        self._splitter.build(feature_bound, self._random_state)
        
        # Build tree
        self._tree = Tree()
        
        # initialize the root node
        root_id = 1
        self._tree.data_id[root_id] = [np.arange(len(X[i])) for i in range(num_parties)]
        self._tree.node_inf[root_id] = NodeInf(root_id)
        
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
        return self
            
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
        
    
    
        
    
    # check current node    
    def _check_node(self, node_id, X, y):
        num_parties = len(X)
        #data_id_this = [ self._tree.data_id[node_id][party_id] for party_id in range(num_parties) ]
        data_id_this = self._tree.data_id[node_id]
                        
        num_ins_array, num_ins_by_class_array = self._count_instances(data_id_this, y)
        
        #print(num_ins_array)
        #print(num_ins_by_class_array)
        
        # there is no instances in this node
        if num_ins_array is None:
            return False
    
        this_node_gini = self._tree.node_inf[node_id].gini
        if this_node_gini is None:
            this_node_gini = _secure_gini(num_ins_array, num_ins_by_class_array)
            self._tree.node_inf[node_id].gini = this_node_gini
            
        # this node is pure, or there is no split avaliable
        if this_node_gini < 1e-4 or self._splitter.is_empty():
            # the node is a leaf node, so label its major class
            majority_class_idx = _find_majority(num_ins_by_class_array)
            self._tree.node_inf[node_id].majority_class = (majority_class_idx, self._distinct_class_vals[majority_class_idx])
            self._tree.leaf_node.append(node_id)
            return False
        
        
        # otherwise, find the best split and split this node
        
          # successfully split
        if self._find_split(node_id, X, y):
            return True
          # no any existing split point can split this node, so mark it as a leaf
        else:
            majority_class_idx = _find_majority(num_ins_by_class_array)
            self._tree.node_inf[node_id].majority_class = (majority_class_idx, self._distinct_class_vals[majority_class_idx])
            self._tree.leaf_node.append(node_id)
            return False
            
    
    def _parallel_check_split(self, node_id, feature, X, y):
        
        # if self.verbose > 1:
        #     print("check node_id: %d, feature id: %d" % (node_id, feature))
        num_parties = len(X)
        data_id_this = self._tree.data_id[node_id]
        split_candidates = self._splitter.get_split(feature)

        min_gini = np.inf
        best_split = None
        gini_left_split = None
        gini_right_split = None

        for split in split_candidates:
            data_id_left = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] <= split[1])[0]]\
                                 for party_id in range(num_parties)]
            data_id_right = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] > split[1])[0]]\
                                 for party_id in range(num_parties)]
            
            num_ins_array_left, num_ins_by_class_left = self._count_instances(data_id_left, y)
            num_ins_array_right, num_ins_by_class_right = self._count_instances(data_id_right, y)
            if num_ins_array_left is not None and num_ins_array_right is not None:
                gini_left = _secure_gini(num_ins_array_left, num_ins_by_class_left)
                gini_right = _secure_gini(num_ins_array_right, num_ins_by_class_right)
                if gini_left + gini_right < min_gini:
                    min_gini = gini_left + gini_right
                    best_split = split
                    gini_left_split = gini_left
                    gini_right_split = gini_right
        
        return (min_gini, feature, best_split, gini_left_split, gini_right_split)

    def _parallel_check_feature_split(self, node_id, feature_split, X, y):
        
        feature, split = feature_split
        # if self.verbose > 1:
        #     print("check node_id: %d, feature id: %d, split id: %d" % (node_id, feature, split[0]))
        num_parties = len(X)
        data_id_this = self._tree.data_id[node_id]
        #split_candidates = self._splitter.get_split(feature)

        min_gini = np.inf
        gini_left = np.inf
        gini_right = np.inf

        #for split in split_candidates:
        data_id_left = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] <= split[1])[0]]\
                                 for party_id in range(num_parties)]
        data_id_right = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] > split[1])[0]]\
                                 for party_id in range(num_parties)]
            
        num_ins_array_left, num_ins_by_class_left = self._count_instances(data_id_left, y)
        num_ins_array_right, num_ins_by_class_right = self._count_instances(data_id_right, y)
        if num_ins_array_left is not None and num_ins_array_right is not None:
            gini_left = _secure_gini(num_ins_array_left, num_ins_by_class_left)
            gini_right = _secure_gini(num_ins_array_right, num_ins_by_class_right)
            if gini_left + gini_right < min_gini:
                min_gini = gini_left + gini_right

        
        return min_gini, gini_left, gini_right
    
    def _find_split(self, node_id, X, y):    
        num_parties = len(X)
        data_id_this = self._tree.data_id[node_id]
        visiting_features = []
        if self.max_features_ < self.n_features_:
            visiting_features = self._random_state.choice(
                                  list(range(self.n_features_)), self.max_features_, replace=False)
        else:
            visiting_features = list(range(self.n_features_))
            
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
        

        for result in best_split_of_allfeatures:
            if result[0] < min_gini:
                min_gini, best_feature, best_split, gini_left_split, gini_right_split = result
        
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

        
        
        data_id_left_split = [data_id_this[party_id][
                        np.where(X[party_id][data_id_this[party_id]][:, best_feature] <= best_split[1])[0]]\
                             for party_id in range(num_parties)]
        data_id_right_split = [data_id_this[party_id][
                        np.where(X[party_id][data_id_this[party_id]][:, best_feature] > best_split[1])[0]]\
                             for party_id in range(num_parties)]
    


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
        if min_gini == np.inf:
            return False
            
        self._tree.node_inf[node_id].set_split_info(best_feature, best_split, min_gini)
        
        left_child_id = node_id * 2
        self._tree.data_id[left_child_id] = data_id_left_split
        self._tree.node_inf[left_child_id] = NodeInf(left_child_id, gini_left_split)
        
        right_child_id = left_child_id + 1
        self._tree.data_id[right_child_id] = data_id_right_split
        self._tree.node_inf[right_child_id] = NodeInf(right_child_id, gini_right_split)
        
        self._splitter.take_split(best_feature, best_split)
        
        if self.verbose > 1:
            print("split node id: {0:d}\n"
                  "feature id: {1:d}\n"
                  "split: ({2:d},{3:f})\n"
                  "gini: {4:f}\n".format(node_id, best_feature, best_split[0], best_split[1], min_gini))
        
        return True

      
    def _count_instances(self, data_id, y):
        num_parties = len(y)
        
        # count the total number of instances of each party locally
        num_ins_array = np.array([ len(data_id[party_id]) \
                         for party_id in range(num_parties) ]).reshape(1,-1)
        
        basci_operator = bbb(num_parties)
        # secure determine if the total number of instances within current node is zero
        if basci_operator.Output(basci_operator.EQZ(num_ins_array, 32))[0,0]:
            return None, None
        
        # count the number of instances of each class locally
        num_ins_by_class = [Counter(y[party_id][data_id[party_id]]) \
                for party_id in range(num_parties)]        
        
        # for the sake of computation, transform class labels in self._distinct_class_vals to 0, 1, 2, 3
        num_ins_by_class_array = np.zeros( (len(self._distinct_class_vals), num_parties), dtype='int')
        class_series = 0
        for class_id in self._distinct_class_vals:
            for party_id in range(num_parties):
                num_ins_by_class_array[class_series][party_id] = num_ins_by_class[party_id][class_id]
            
            class_series += 1
        
        return num_ins_array, num_ins_by_class_array
        
class splitter(list):
    """ A simple class to model a splitter used for federated decision tree building
    
    """
    def __init__(self, n_features, n_splits):
        self.n_features = n_features
        self.n_splits = n_splits
        super().__init__([]) 
        self.split_index = np.ones((n_features, n_splits), dtype=bool)
        self._is_bulit = False
        
    def build(self, feature_bound, random_state):
        if len(feature_bound) != self.n_features:
            raise ValueError("Could not build splitter with %d feature bounds but %d features." %\
                                 (len(feature_bound), self.n_features))
        for bound in feature_bound:
            self.append(random_state.random(self.n_splits) * (bound[1] - bound[0]) + bound[0])
        self._is_bulit = True
        
        return None
    
    def rebuild(self, feature_bound, random_state):
        """ Rebulid the splitter by resampling the values for splitting while keeping the records in split_index unchanged
            This method is used when all existing splits cannot split the current node that has been checked to be impure.
        
        """
        if self._is_bulit is False:
            raise AttributeError("this splitter instance has not been bulit, call 'build' method at first.")
        self.clear()
        for bound in feature_bound:
            self.append(random_state.random(self.n_splits) * (bound[1] - bound[0]) + bound[0])
        
        return None
        
    def take_split(self, feature_id, split):
        """ Label the split used so that it is not checked afterwards
        
            Parameters
            ---------------------------------------
            feature_id
            split = (split_id, split_value)
        """
        if self._is_bulit is False:
            raise AttributeError("this splitter instance has not been bulit, call 'build' method at first.")
        self.split_index[feature_id, split[0]] = False
        return None
    
    def get_split(self, feature_id):
        """ get splits haven't been taken
        
        """
        
        if self._is_bulit is False:
            raise AttributeError("this splitter instance has not been bulit, call 'build' method at first.")
        split_id = np.where(self.split_index[feature_id] == True)[0]
        split = [(i, self[feature_id][i]) for i in split_id]
        return split
    
    def is_empty(self):
        """ The splitter is empty if and only if the splits of all features have been taken
        """
        if self._is_bulit is False:
            raise AttributeError("this splitter instance has not been bulit, call 'build' method at first.")
        
        return not self.split_index.any()


class Tree():
    """ A simple class to model a federated tree with associated information
        
        Use the node_id of a tree as entry to record the node information
        
        eg.             
                          1
                        /   \
                       2     3
                      / \   
                     4   5
                        / \ 
                       10 11 
                       
    """
    
    def __init__(self):
        #self.feature_id = [-1 for _ in range(n_features)]
        # Use node_id as entry to record the information and data_id of each node
        self.node_inf = {}
        self.data_id = {}
        #self.end_node = -1
        self.leaf_node = []
        
        
    def create_decision_condition(self):
        pass
    
    def predict(self, X):
        pass
        
        
        
    def apply(self, X):
        """Return the indices of the leaf that each sample in X is predicted as.
        
        Parameters:
        -----------------------------
        X：Array-like of shape (n_samples, n_features)
           Unlike the fit method where the input is federated, here X is located in the initiator. The initiator uses the
           learned tree for inference. This corresponds to the setting of our framework. 
           
        Returns:
        -----------------------------
        X_leaves: Array-like of shape(n_samples,)
            For each data point x in X, return the index of the leaf x ends up in. 
            Leaves are numbered with the indice of self.leaf_node 
        
        """
        X_leafs = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            j = 1
            while(self.node_inf[j].majority_class is None):
                if X[i, self.node_inf[j].best_feature_id] <= self.node_inf[j].best_split[1]:
                    j *= 2
                else:
                    j = j * 2 + 1
            X_leafs[i] = j
            
        return X_leafs
    
class NodeInf():
    """ A simple class to model a node of the tree with associated information
    
    """
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
        
    def set_split_info(self, best_feature_id, best_split, min_gini_split):
        self.best_feature_id = best_feature_id
        self.best_split = best_split
        self.min_gini_split = min_gini_split



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
        
        random_state = check_random_state(self.random_state)
        
        # no need each party to have all class labels
        self._distinct_class_vals = np.unique(np.concatenate([y[party_id] for party_id in range(num_parties)]))
        
        self._validate_estimator()
        trees = [self._make_estimator(append=False,
                                      random_state=random_state)
                for i in range(self.n_estimators)]
        
        self.estimators_ = []
        
        trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_trees)(
                t, X, y, i, len(trees), self.verbose)
            for i, t in enumerate(trees))
        
        self.estimators_.extend(trees)
        
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
        y_pred_idx = np.empty(all_proba.shape[0], dtype=self._distinct_class_vals.dtype)
        for i in range(len(y_pred)):
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
        