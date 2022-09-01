
""" Basics for implementing a federated decision tree classifier based on a secure multi-parties float-point computation protocol 

    All classes and methods in this module cannot be used directly.

"""



__author__ = "Zhiyu Liang"


# import sys

# sys.path.append('../')



# from numba import jit

import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator

from FedHC.secret_sharing_based_computation._sfc import fl
from FedHC.secret_sharing_based_computation._basic_building_blocks import basic_building_blocks as bbb

from collections import Counter

from FedHC import glb




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


def secure_F_stat(distance_collection):
    """ Compute F-tatistic shapelet quality measure with secure float point number computation protocol
    
    Parameter
    -----------------------------------------
    distance_collection      : list of num_parties length. Each party stores a dict taking class_id
                                as entry and returing a list of the masked distance of that class 
    
    """
    num_parties = len(distance_collection)
    
    # Make the total number of distances public
    total_num_distances = sum([len(val) for party in distance_collection \
                                       for val in party.values()])
            
    # Each party holds the whole label space thus knows the number of distinct class values
    num_classes = len(distance_collection[0])
            
    # Each party aggregates the distance by class and totally
    distance_sum_by_class = [{class_id: sum(class_distance) \
                                                for class_id, class_distance in party.items()} \
                                     for party in distance_collection]
    
    num_distance_by_class = [{class_id: len(class_distance) \
                                                for class_id, class_distance in party.items()} \
                                     for party in distance_collection]
    
    distance_sum_total = [sum(party.values()) \
                            for party in distance_sum_by_class]
    
    # Each party compute a share of the total average distance
    dis_avg_total = np.array([distance_sum_total]) / total_num_distances
    
    
    FL = fl(9, 23, num_parties)
    basci_operator = bbb(num_parties)
    # Compute sum (D_i - D)^2 by sum (D_i)^2 - 2 * D_i * D + D * D, where D_i is public while D is shared
    # Firstly, compute D * D with Beaver's multiplication triplet
    variability_between_group = basci_operator.Mul(dis_avg_total, dis_avg_total)
    variability_between_group *= num_classes
    
    # Each party compute a share of sum sum (d_j - D_i)^2   
    variability_within_group = np.zeros((1, num_parties))
    
    # Now compute D_i for each class i with secure floating point number division
    # and make D_i public, then complete the computation of variability between and within group
    #dis_avg_by_class = {}
    for class_id in distance_collection[0].keys():
        # Access the aggregated distance and the count of the class
        distance_sum = np.array([party_distance[class_id] for party_distance in distance_sum_by_class])
        num_distance = np.array([party_num[class_id] for party_num in num_distance_by_class])
        
        # Each party transform the distance to floating point number and shares with others
        distance_sum_shared = FL.real_to_fl(distance_sum)
        # The count number can be seen as shared integer, thus is directly transformed into floating point
        # format
        num_distance_shared = FL.Int2FL(num_distance, 32, 23)
        
        dis_avg, error = FL.FLDiv(FL.FLSum(distance_sum_shared), num_distance_shared)
        # if basci_operator.Output(error)[0,0] == 1:
        #     raise ValueError('Division by zero, num_distance_shared is %f' \
        #                      % FL.fl_to_real(num_distance_shared))                
        
        # Open the average distance of this class
        dis_avg = FL.fl_to_real(dis_avg)[0,0]
        #dis_avg_by_class[class_id] = dis_avg
        
        # Each party minus a share of 2 * D_i * D while only one of the party plus D_i ^2
        variability_between_group -= 2 * dis_avg * dis_avg_total
        variability_between_group[:, 0] += dis_avg * dis_avg
        
        # Each party aggregates (d_j - D_i)^2 for all d_j locally
        variability_within_group += np.array([np.sum(np.array(party_distance[class_id]) - dis_avg) ** 2 \
                                             for party_distance in distance_collection]).reshape(1, -1)
    #print(variability_between_group, variability_within_group)
    F_stat, error = FL.FLDiv(FL.FLSum(FL.real_to_fl(variability_between_group.reshape(-1,1))), \
                              FL.FLSum(FL.real_to_fl(variability_within_group.reshape(-1,1))))
    
    # if basci_operator.Output(error)[0,0] == 1:
    #     raise ValueError('Division by zero, variability_within_group is %f' \
    #                          % FL.fl_to_real(variability_within_group))
    # All participants sends their shares to the initiator to open F_stat
    result = FL.fl_to_real(F_stat)[0][0]
    # get the final F_stat
    result = result * (total_num_distances - num_classes) / (num_classes - 1)
    return result

class BaseFederatedDecisionTree(BaseEstimator):
 
    
    def fit(self, X, y):
        pass
            
    def apply(self, X):
        pass
        
        
    
    
    def predict(self, X):
        pass
        
    
    
        
    
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
        #if this_node_gini < 1e-4 or self._splitter.is_empty():
        if this_node_gini < 1e-4:    
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

        glb._init()
        glb.set_value('communication', 0)

        data_id_this = self._tree.data_id[node_id]
        #split_candidates = self._splitter.get_split(feature)
        bound = self._tree.feature_bound[node_id][feature]
        split_candidates = self._random_state.random(self.n_splits) * (bound[1] - bound[0]) + bound[0]

        min_gini = np.inf
        best_split = None
        gini_left_split = None
        gini_right_split = None

        for split in split_candidates:
            data_id_left = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] <= split)[0]]\
                                 for party_id in range(num_parties)]
            data_id_right = [data_id_this[party_id][
                            np.where(X[party_id][data_id_this[party_id]][:, feature] > split)[0]]\
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

                    # no need to check extra splits if current split is good enough
                    if min_gini < 1e-4:
                        break
        
        return (min_gini, feature, best_split, gini_left_split, gini_right_split), glb.get_value('communication')

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
    


    def _parallel_check_feature(self, node_id, feature, X, y):
        """ check the F-stat of each feature
        """
        num_parties = len(y)
        glb._init()
        glb.set_value('communication', 0)

        data_id_this = self._tree.data_id[node_id]

        feature_collection = [{class_val:\
                                X[party_id][np.intersect1d(np.where(y[party_id] == class_val)[0], data_id_this[party_id])][:, feature]
                                            for class_val in self._distinct_class_vals} 
                                for party_id in range(num_parties)]
        
        F_stat = secure_F_stat(feature_collection)
        
        return F_stat, glb.get_value('communication')

      
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



class base_splitter(list):
    """ A simple class to model a splitter used for federated decision tree building
    
    """
      
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


class base_Tree():
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
                if X[i, self.node_inf[j].best_feature_id] <= self.node_inf[j].best_split:
                    j *= 2
                else:
                    j = j * 2 + 1
            X_leafs[i] = j
          
        return X_leafs
  
class base_NodeInf():
    """ A simple class to model a node of the tree with associated information
  
    """
    
      
    def set_split_info(self, best_feature_id, best_split, min_gini_split):
        self.best_feature_id = best_feature_id
        self.best_split = best_split
        self.min_gini_split = min_gini_split