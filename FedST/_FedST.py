""" A local implementation of a federated shapelet transform method based on a secure multi-parties float-point computation protocol """

__author__ = "Zhiyu Liang"

# import sys

# sys.path.append('../')

import heapq
import os
import time
import warnings
from itertools import zip_longest
from operator import itemgetter


import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution
from sktime.transformations.base import \
    _PanelToTabularTransformer
from FedHC.secret_sharing_based_computation import fl
from FedHC.secret_sharing_based_computation import basic_building_blocks as bbb

from FedHC import glb

warnings.filterwarnings("ignore", category=FutureWarning)


class FederatedShapeletTransform(_PanelToTabularTransformer):
    """ Federated shapelet transform
    
    
    Parameters
    -------------------------------------------------
    min_shapelet_length                 : int, lower bound on candidate
    shapelet lengths (default = 3)
    max_shapelet_length                 : int, upper bound on candidate
    shapelet lengths (default = inf or series length)
    max_shapelets_to_store              : int, upper bound on number of
    shapelets to retain                    (default = 200 * num_classes)
    random_state                        : RandomState, int, or none: to
    control random state objects for deterministic results (default = None)
    verbose                             : int, level of output printed to
    the console (for information only) (default = 0)
    remove_self_similar                 : boolean, remove overlapping
    "self-similar" shapelets from the final transform (default = True)
    
    
    """
    
    def __init__(self, 
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 max_shapelets_to_store=200,
                 random_state=None,
                 verbose=0,
                 remove_self_similar=True):
        
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.max_shapelets_to_store = max_shapelets_to_store
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_similar = remove_self_similar
        self.predefined_F_stat_rejection_level = 0
        self.shapelets = None
        
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """ A method to fit the shapelet transform to a federated X and y
        
        
        Parameters
        ---------------------------
        X: list of pandas DataFrame
            The training samples of party A, B, C, ...
        y: list of array-like
            The class values for X
            
        Returns
        ---------------------------
        self: FederatedShapeletTransform
            This estimator
        """
        
        num_parties = len(X)
        num_ins = [len(y[i]) for i in range(len(y))]

        glb._init()
        glb.set_value('communication', 0)
        
        X = [np.array(
            [[X[i].iloc[r, c].values for c in range(len(X[i].columns))] for r in
             range(len(X[i]))]) for i in range(num_parties)]
        X_lens = [np.repeat(X[i].shape[-1], X[i].shape[0]) for i in range(num_parties)]
        
        candidates_evaluated = 0
        
        # each party has the all label values
        # distinct_class_vals = class_distribution(np.asarray(y[0]).reshape(-1, 1))[0][0]
        
        # no need each party to have all class labels
        distinct_class_vals = np.unique(np.concatenate([y[party_id] for party_id in range(num_parties)]))

        self.max_shapelets_to_store *= len(distinct_class_vals)
        
        candidates_evaluated = 0
        
        # The number of series within the initiator
        # Those series are used for candidates generation
        num_series_to_visit = num_ins[0]
        
        shapelet_heap = ShapeletPQ()
        
        self._random_state = check_random_state(self.random_state)
        
        def _round_robin(*iterables):
            sentinel = object()
            return (
                a
                for x in zip_longest(*iterables, fillvalue=sentinel)
                for a in x
                if a != sentinel
            )
        
        # party 0 is the initiator, only of which the shapelet candidates are checked
        case_ids_by_class = {i: np.where(y[0] == i)[0] for i in distinct_class_vals}
        
        if (type(self) is ContractedFederatedShapeletTransform):
            for i in range(len(distinct_class_vals)):
                self._random_state.shuffle(case_ids_by_class[distinct_class_vals[i]])

        num_train_per_class = {i: len(case_ids_by_class[i]) for i in case_ids_by_class}
        round_robin_case_order = _round_robin(
            *[list(v) for k, v in case_ids_by_class.items()]
        )
        # party 0 is the initiator, only of which the shapelet candidates are checked
        cases_to_visit = [(i, y[0][i]) for i in round_robin_case_order]
        
        possible_candidates_per_series_length = {}
        
        # At each iteration, self.num_candidates_per_case candidates of each case are checked
        # At the end of each iteration, if time is remaining, a new iteration begins
        # The checked candidates are recorded which cann't be chosen at the following iterations
        # The search ends up when either all candidates are checked or the time is up   
        candidates_checked = [ [] for _ in range(len(cases_to_visit))]
        # a flag to indicate if all cases of the training series have been checked
        all_cases_checked = np.array([False for _ in range(len(cases_to_visit))])
        
        # cases in other parties are used to score the shapelet candidates
        # access by party index
        cases_to_compare = {idx: [(i, y[idx][i]) for i in range(num_ins[idx])] \
                            for idx in range(num_parties) if idx != 0}
        
        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()

        def time_taken():
            return time.time() - start_time
        
        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()
        
        # for every series
        case_idx = 0
        
        
        
        while case_idx < len(cases_to_visit):
            # skip if all candidates of this case have been checked
            if all_cases_checked[case_idx]:
                case_idx = 0 if case_idx >= num_series_to_visit else case_idx + 1
                continue
            
            series_id = cases_to_visit[case_idx][0]
            this_class_val = cases_to_visit[case_idx][1]
            
            if self.verbose:
                    print(  # noqa
                        "visiting series: "
                        + str(series_id)
                        + " (#"
                        + str(case_idx + 1)
                        + ")"
                    )
                    
            this_series_len = len(X[0][series_id][0])
            
            # The bound on possible shapelet lengths will differ
            # series-to-series if using unequal length data.
            # However, shapelets cannot be longer than the series, so set to
            # the minimum of the series length
            # and max shapelet length (which is inf by default)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(
                    this_series_len, self.max_shapelet_length
                )
            
            # all possible start and lengths for shapelets within this
            # series (calculates if series length is new, a simple look-up
            # if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series
            # of the same length before.
            candidate_starts_and_lens = possible_candidates_per_series_length.get(
                this_series_len
            )
            # else calculate them for this series length and store for
            # possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length]
                    for start in range(
                        0, this_series_len - self.min_shapelet_length + 1
                    )
                    for length in range(
                        self.min_shapelet_length, this_shapelet_length_upper_bound + 1
                    )
                    if start + length <= this_series_len
                ]
                possible_candidates_per_series_length[
                    this_series_len
                ] = candidate_starts_and_lens
            
            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)
            
            # limit search otherwise:
            if hasattr(self, "num_candidates_to_sample_per_case"):
                num_candidates_per_case -= len(candidates_checked[case_idx])
    
                num_candidates_per_case = min(
                    self.num_candidates_to_sample_per_case, num_candidates_per_case
                )
#                 cand_idx = list(
#                     self._random_state.choice(
#                         list(range(0, len(candidate_starts_and_lens))),
#                         num_candidates_per_case,
#                         replace=False,
#                     )
#                 )
                cand_idx = list(
                      self._random_state.choice(
                          [i for i in range(len(candidate_starts_and_lens)) \
                           if i not in candidates_checked[case_idx]],
                          num_candidates_per_case,
                          replace=False,
                    )
                )
                candidates_to_visit = [candidate_starts_and_lens[x] for x in cand_idx]
                
            for candidate_idx in range(num_candidates_per_case):
                # if shapelet heap is not full yet, set entry
                # criteria to be the predetermined F-stat threshold
                F_stat_cutoff = self.predefined_F_stat_rejection_level
                # otherwise if we have max shapelets already, set the
                # threshold as the F-stat of the current 'worst' shapelet we have
                if shapelet_heap.get_size() >= self.max_shapelets_to_store:
                    F_stat_cutoff = max(shapelet_heap.peek()[0],F_stat_cutoff)
                
                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]
                
                candidate = FederatedShapeletTransform.zscore(
                    X[0][series_id][:, cand_start_pos : cand_start_pos + cand_len]
                )
                
                
                # now go through all other series and get a distance from
                # the candidate to each
                # Each party records the distances from current candidate to its series
                distance_collection = [{class_id: [] for class_id \
                                                  in distinct_class_vals} \
                                       for party_id in range(num_parties)]
                
                candidate_rejected = False
                
                # The initiator generates a random mask to mask all the distance
                # thus any participant cannot see the real distance but can aggregate 
                # the distance correctly
                mask_initiator = np.random.random() * cand_len
                
                #class_identifier = {}
                # for each shapelet candidate, reshuffle the series indices showning to the initiator
                comparison_series_idx_reshuffed = {comparison_party: \
                                                   list(range(len(cases_to_compare[comparison_party]))) \
                                                   for comparison_party in range(1, num_parties)}
                for comparison_party in range(1, num_parties):
                    self._random_state.shuffle(comparison_series_idx_reshuffed[comparison_party])
                
            
                
                for comparison_party in range(num_parties):
                    if comparison_party == 0:
                        for comparison_series_idx in range(len(cases_to_visit)):
                            i = cases_to_visit[comparison_series_idx][0]
                            
                            if y[0][i] != cases_to_visit[comparison_series_idx][1]:
                                raise ValueError("class match sanity test broken")
                            if i == series_id:
                                # don't evaluate candidate against own series
                                continue
                            
                            bsf_dist = np.inf

                            start_left = cand_start_pos
                            start_right = cand_start_pos + 1

                            if X_lens[0][i] == cand_len:
                                start_left = 0
                                start_right = 0

                            for num_cals in range(max(1, int(np.ceil((X_lens[0][i] -
                                                              cand_len) /
                                                             2)))):  # max
                                # used to force iteration where series len ==
                                # candidate len
                                if start_left < 0:
                                    start_left = X_lens[0][i] - cand_len

                                comparison = FederatedShapeletTransform.zscore(
                                X[0][i][:, start_left: start_left + cand_len])
                                dist_left = np.linalg.norm(candidate - comparison)
                                bsf_dist = min(dist_left * dist_left, bsf_dist)

                                # for odd lengths
                                if start_left == start_right:
                                    continue

                                # right
                                if start_right == X_lens[0][i] - cand_len + 1:
                                    start_right = 0
                                comparison = FederatedShapeletTransform.zscore(
                                    X[0][i][:, start_right: start_right + cand_len])
                                dist_right = np.linalg.norm(candidate - comparison)
                                bsf_dist = min(dist_right * dist_right, bsf_dist)

                                start_left -= 1
                                start_right += 1
                            
                            
                            distance_collection[0][y[0][i]].append(bsf_dist + mask_initiator)
                            
                    # 2PC to compute distance through secure dot-product
                    else:
                        for comparison_series_idx in range(len(comparison_series_idx_reshuffed[comparison_party])):
                            # Mapping the series index shown in the initiator to the index of the participant
                            comparison_series_idx_participant = comparison_series_idx_reshuffed[ \
                                                                    comparison_party][comparison_series_idx]
                            
                            # The participant does
                            i = cases_to_compare[comparison_party][comparison_series_idx_participant][0]
                            
                            if y[comparison_party][i] != cases_to_compare[ \
                                                            comparison_party][ \
                                                            comparison_series_idx_participant][1]:
                                raise ValueError("class match sanity test broken")
                            
                            start_point_all = list(range(X_lens[comparison_party][i] - cand_len + 1))
                            # The participant shuffles the subsequences so the initiator can't see 
                            # the location of each subsequence
                            self._random_state.shuffle(start_point_all)
                            
                            # The participants generates a random mask to mask the distance from the candidate
                            # to each of the subsequence of the comparison series 
                            # The initiator cannot see the real distance but can find the minimum distance
                            # correctly with every distance shifted equally
                            mask_participant = np.random.random() * cand_len
                            
                            # The initiator does
                            bsf_dist = np.inf
                            candidate_square_sum = np.linalg.norm(candidate) ** 2
             
                            # The initiator does under the assistance of the participant
                            for num_cals in range(len(start_point_all)):
                                start_point = start_point_all[num_cals]
                                comparison = FederatedShapeletTransform.zscore(
                                    X[comparison_party][i][:, start_point: start_point + cand_len])
                                alpha, beta = FederatedShapeletTransform.secure_dot_product(candidate, comparison)
                                
                                # The distance of x and y is x^2 + y^2 - 2 * dot_product(x, y)
                                # The participants sends x^2 + beta and a random mask
                                # to the initiator to open the masked dot product and distance
                                # The initiator cannot see the real distance but can find the minimum distance
                                # correctly with every distance shifted mask
                                masked_dist = np.linalg.norm(comparison) ** 2 + \
                                                candidate_square_sum - \
                                                2 * (beta - np.sum(alpha)) + mask_participant
                                bsf_dist = min(bsf_dist, masked_dist)
                            
                            # The initiator sends the masked distance to the participant
                            bsf_dist += mask_initiator
                            # The participant remove the the mask generated by itself
                            # and store the remain, as well as the class value of the comparison series
                            distance_collection[comparison_party][y[comparison_party][i]].append( \
                                                                            bsf_dist - mask_participant)
                    
                # Calculate the F-stat
                F_stat = FederatedShapeletTransform.secure_F_stat(distance_collection)
                
                candidates_evaluated += 1
                if self.verbose > 3 and candidates_evaluated % 100 == 0:
                    print("candidates evaluated: " + str(candidates_evaluated))
                
                accepted_candidate = Shapelet(series_id, cand_start_pos,
                                                  cand_len, F_stat,
                                                  candidate)
                
                # add to min heap to store shapelet
                shapelet_heap.push(accepted_candidate)
                
                # Takes into account the use of the MAX shapelet calculation
                # time to not exceed the time_limit (not exact, but likely a
                # good guess).
                if hasattr(self,
                           'time_contract_in_mins') and \
                        self.time_contract_in_mins \
                        > 0:
                    time_now = time_taken()
                    time_this_shapelet = (time_now - time_last_shapelet)
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                        print('max_time_calc_shapelet: ', max_time_calc_shapelet)
                    time_last_shapelet = time_now
                    # add a little 1% leeway to the timing incase one run was slightly faster than
                    # another based on the CPU.
                    time_in_seconds = self.time_contract_in_mins * 60
                    max_shapelet_time_percentage = (
                        max_time_calc_shapelet / 100.0) * 0.75
                    if (time_now + max_shapelet_time_percentage) > \
                            time_in_seconds:
                        if self.verbose > 0:
                            print(
                                "No more time available! It's been {0:02d}:{"
                                "1:02}".format(
                                    int(round(time_now / 60, 3)), int((round(
                                        time_now / 60, 3) - int(
                                        round(time_now / 60, 3))) * 60)))
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            if candidate_rejected is False:
                                print(
                                    "Candidate finished. {0:02d}:{1:02} "
                                    "remaining".format(
                                        int(round(
                                            self.time_contract_in_mins -
                                            time_now / 60,
                                            3)),
                                        int((round(
                                            self.time_contract_in_mins -
                                            time_now / 60,
                                            3) - int(
                                            round((self.time_contract_in_mins
                                                   - time_now) / 60, 3))) *
                                            60)))
                            else:
                                print(
                                    "Candidate rejected. {0:02d}:{1:02} "
                                    "remaining".format(int(round(
                                        (self.time_contract_in_mins -
                                         time_now) / 60, 3)),
                                        int((round(
                                            (self.time_contract_in_mins -
                                             time_now) / 60,
                                            3) - int(
                                            round((self.time_contract_in_mins -
                                                   time_now) / 60, 3))) * 60)))

            # stopping condition: in case of iterative transform (i.e.
            # num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time
            #                     limit has been reached)
            
            candidates_checked[case_idx].extend(cand_idx)
            if len(candidates_checked[case_idx]) == len(candidate_starts_and_lens):
                all_cases_checked[case_idx] = True
            
            if self.verbose > 0:
                print(
                      "Case %2d: %2d/%2d candidates have been checked" % 
                      (case_idx, len(candidates_checked[case_idx]), len(candidate_starts_and_lens)))
                
                print("candidates checked:\n", candidates_checked[case_idx])
            
            
            case_idx += 1
            
            if case_idx >= num_series_to_visit:
                if hasattr(self,
                           'time_contract_in_mins') and time_finished is not \
                        True:
                    case_idx = 0
            elif case_idx >= num_series_to_visit or time_finished or all_cases_checked.all():
                if self.verbose > 0:
                    print("Stopping search")
                break        
                                
        # remove self similar here
        
        # get list of shapelets
        # sort by quality
        # remove self similar                   
                                
        shapelets_descending_quality = sorted(shapelet_heap.get_array(), key=itemgetter(0), reverse=True)
        
        if self.remove_self_similar and len(shapelets_descending_quality) > 0:
            shapelets_descending_quality = FederatedShapeletTransform.remove_self_similar_shapelets(\
                                                            shapelets_descending_quality)
        
        else:
            # shapelets_descending_quality[i] is a tuple with (quality,id,Shapelet),
            # so we need to access [2]
            shapelets_descending_quality = [x[2] for x in shapelets_descending_quality]
        
        # if we have more than max_shapelet, trim to that
        # amount here
        if len(shapelets_descending_quality) > self.max_shapelets_to_store:
            max_n = self.max_shapelets_to_store
            shapelets_descending_quality = shapelets_descending_quality[:max_n]
        
        self.shapelets = shapelets_descending_quality
        self.is_fitted_ = True
        
        # warn the user if fit did not produce any valid shapelets
        if len(self.shapelets) == 0:
            warnings.warn(
                "No valid shapelets were extracted from this dataset and "
                "calling the transform method "
                "will raise an Exception. Please re-fit the transform with "
                "other data and/or "
                "parameter options.")

        self.communication_amount = glb.get_value('communication')
        self._is_fitted = True
        return self                    
    
    # Transform a set of local data into distances to each extracted shapelet.
    # This method is used to transform the testing data.
    def transform_locally(self, X, y=None, **transform_params):
        """Transform local X according to the extracted shapelets (self.shapelets)
        
        Parameters
        ----------------
        X: Pandas DataFrame
            The input dataframe of this party(usually the initiator) to transform
            
        
        Returns
        -----------------
        output: DataFrame
            The transformed dataframe of this party in tabular format
        
        """
    
        self.check_is_fitted()

        if len(self.shapelets) == 0:
            raise RuntimeError(
                "No shapelets were extracted in fit that exceeded the "
                "minimum information gain threshold. Please retry with other "
                "data and/or parameter settings.")


        _X = np.array(
            [[X.iloc[r, c].values for c in range(len(X.columns))]
             for r in range(len(X))])

        output = np.zeros((len(_X), len(self.shapelets)))

        for i in range(0, len(_X)):
            this_series = _X[i]
                    
                    
            # get the s^th shapelet
            for s in range(0, len(self.shapelets)):
                # find distance between this series and each shapelet
                min_dist = np.inf
                this_shapelet_length = self.shapelets[s].length
                start_left = self.shapelets[s].start_pos
                start_right = start_left + 1

                if len(this_series[0]) == this_shapelet_length:
                    start_left = 0
                    start_right = 0
                        
                        
                for num_cals in range(max(1, int(np.ceil((len(this_series[0]) -
                                                            this_shapelet_length) /
                                                            2)))):  
                    # max used to force iteration where series len == shapelet length

                    if start_left < 0:
                        start_left = len(this_series[0]) - this_shapelet_length

                    comparison = FederatedShapeletTransform.zscore(
                                                            this_series[:, start_left:(start_left +
                                                            this_shapelet_length)])

                    dist_left = np.linalg.norm(self.shapelets[s].data - comparison)
                    dist_left = dist_left * dist_left
                    dist_left = 1.0 / this_shapelet_length * dist_left
                    min_dist = min(min_dist, dist_left)

                    # for odd lengths
                    if start_left == start_right:
                        continue

                    # right
                    if start_right == len(this_series[0]) - this_shapelet_length + 1:
                        start_right = 0
                    comparison = FederatedShapeletTransform.zscore(
                                this_series[:, start_right: start_right + this_shapelet_length])
                    dist_right = np.linalg.norm(self.shapelets[s].data - comparison)
                    dist_right = dist_right * dist_right
                    dist_right = 1.0 / this_shapelet_length * dist_right
                    min_dist = min(dist_right, min_dist)

                    start_left -= 1
                    start_right += 1

                output[i][s] = min_dist + self.mask_initiator[s]

        return output 

    # Transform a set of federated data into distances to each extracted shapelet
    def transform(self, X, y=None, **transform_params):
        """Transform federated X according to the extracted shapelets (self.shapelets)
        
        Parameters
        ----------------
        X: list of pandas DataFrame
            The input dataframe of each party to transform
            
        
        Returns
        -----------------
        output: list of DataFrame
            The transformed dataframe of each party in tabular format
        
        """
        
        self.check_is_fitted()
        
        if len(self.shapelets) == 0:
            raise RuntimeError(
                "No shapelets were extracted in fit that exceeded the "
                "minimum information gain threshold. Please retry with other "
                "data and/or parameter settings.")
        
        num_parties = len(X)
        
        _X = [np.array(
            [[X[party_id].iloc[r, c].values for c in range(len(X[party_id].columns))]
             for r in range(len(X[party_id]))]) for party_id in range(num_parties)]
        
        output = [np.zeros((len(_X[party_id]), len(self.shapelets))) \
                  for party_id in range(num_parties)]
        
        # Mask each transformed feature with a different random value
        self.mask_initiator = np.random.random(len(self.shapelets)) * \
                                            np.array([s.length for s in self.shapelets])
        
        for party_id in range(num_parties):
            if party_id == 0:
                # for the i^th series to transform
                for i in range(0, len(_X[0])):
                    this_series = _X[0][i]
                    
                    
                    # get the s^th shapelet
                    for s in range(0, len(self.shapelets)):
                        # find distance between this series and each shapelet
                        min_dist = np.inf
                        this_shapelet_length = self.shapelets[s].length
                        start_left = self.shapelets[s].start_pos
                        start_right = start_left + 1

                        if len(this_series[0]) == this_shapelet_length:
                            start_left = 0
                            start_right = 0
                        
                        
                        for num_cals in range(max(1, int(np.ceil((len(this_series[0]) -
                                                                  this_shapelet_length) /
                                                                 2)))):  
                            # max used to force iteration where series len == shapelet length

                            if start_left < 0:
                                start_left = len(this_series[0]) - this_shapelet_length

                            comparison = FederatedShapeletTransform.zscore(
                                                            this_series[:, start_left:(start_left +
                                                            this_shapelet_length)])

                            dist_left = np.linalg.norm(self.shapelets[s].data - comparison)
                            dist_left = dist_left * dist_left
                            dist_left = 1.0 / this_shapelet_length * dist_left
                            min_dist = min(min_dist, dist_left)

                            # for odd lengths
                            if start_left == start_right:
                                continue

                            # right
                            if start_right == len(this_series[0]) - this_shapelet_length + 1:
                                start_right = 0
                            comparison = FederatedShapeletTransform.zscore(
                                this_series[:, start_right: start_right + this_shapelet_length])
                            dist_right = np.linalg.norm(self.shapelets[s].data - comparison)
                            dist_right = dist_right * dist_right
                            dist_right = 1.0 / this_shapelet_length * dist_right
                            min_dist = min(dist_right, min_dist)

                            start_left -= 1
                            start_right += 1

                        output[0][i][s] = min_dist + self.mask_initiator[s]
            
            else:
                # for each shapelet, reshuffle the order of the series to enhance security
                for s in range(0, len(self.shapelets)):
                    this_shapelet_length = self.shapelets[s].length
                    
                    series_id_reshuffled = [i for i in range(len(_X[party_id]))]
                    self._random_state.shuffle(series_id_reshuffled)
                    
                    for idx in range(0, len(_X[party_id])):
                        # The participant accesses the series, reshuffle the start position
                        # and generates the mask
                        i = series_id_reshuffled[idx]
                        this_series = _X[party_id][i]
                        start_pos_shuffled = list(range(len(this_series[0]) - this_shapelet_length + 1))
                        self._random_state.shuffle(start_pos_shuffled)
                        mask_participant = np.random.random() * this_shapelet_length
                        
                        
                        min_dist = np.inf
                        
                        # 2PC to compute the masked minimun distance with secure dot product
                        for start_pos in range(len(this_series[0]) - this_shapelet_length + 1):
                            start_pos = start_pos_shuffled[start_pos]
                            comparison = FederatedShapeletTransform.zscore(
                                                this_series[:, start_pos:(start_pos+this_shapelet_length)])
                            
                            alpha, beta = FederatedShapeletTransform.secure_dot_product(
                                                                    self.shapelets[s].data, comparison)
                                
                            # The distance of x and y is x^2 + y^2 - 2 * dot_product(x, y)
                            # The participants sends x^2 + beta and a random mask
                            # to the initiator to open the masked dot product and distance
                            # The initiator cannot see the real distance but can find the minimum distance
                            # correctly with every distance shifted mask
                            masked_dist = np.linalg.norm(comparison) ** 2 + \
                                          np.linalg.norm(self.shapelets[s].data) ** 2 - \
                                            2 * (beta - np.sum(alpha)) + mask_participant
                            min_dist = min(min_dist, masked_dist)
                        
                        # The participant stores the minimum distance masked by the initiator
                        output[party_id][i][s] = min_dist + self.mask_initiator[s] - mask_participant
        
        self.communication_amount = glb.get_value('communication')

        return output
    
    def get_shapelets(self):
        """An accessor method to return the extracted shapelets

        Returns
        -------
        shapelets: a list of Shapelet objects
        """
        return self.shapelets
    
    @staticmethod
    def remove_self_similar_shapelets(shapelet_list):
        """Remove self-similar shapelets from an input list. Note: this
        method assumes
        that shapelets are pre-sorted in descending order of quality (i.e.
        if two candidates
        are self-similar, the one with the later index will be removed)

        Parameters
        ----------
        shapelet_list: list of Shapelet objects

        Returns
        -------
        shapelet_list: list of Shapelet objects
        """

        # IMPORTANT: it is assumed that shapelets are already in descending
        # order of quality. This is preferable in the fit method as removing
        # self-similar
        # shapelets may be False so the sort needs to happen there in those
        # cases, and avoids a second redundant sort here if it is set to True

        def is_self_similar(shapelet_one, shapelet_two):
            # not self similar if from different series
            if shapelet_one.series_id != shapelet_two.series_id:
                return False

            if (shapelet_one.start_pos >= shapelet_two.start_pos) and (
                    shapelet_one.start_pos <= shapelet_two.start_pos +
                    shapelet_two.length):
                return True
            if (shapelet_two.start_pos >= shapelet_one.start_pos) and (
                    shapelet_two.start_pos <= shapelet_one.start_pos +
                    shapelet_one.length):
                return True

        # [s][2] will be a tuple with (quality,id,Shapelet), so we need to
        # access [2]
        to_return = [shapelet_list[0][2]]  # first shapelet must be ok
        for s in range(1, len(shapelet_list)):
            can_add = True
            for c in range(0, s):
                if is_self_similar(shapelet_list[s][2], shapelet_list[c][2]):
                    can_add = False
                    break
            if can_add:
                to_return.append(shapelet_list[s][2])

        return to_return            
                
                
                
    @staticmethod
    def zscore(a, axis=0, ddof=0):
        """A static method to return the normalised version of series.
        This mirrors the scipy implementation
        with a small difference - rather than allowing /0, the function
        returns output = np.zeroes(len(input)).
        This is to allow for sensible processing of candidate
        shapelets/comparison subseries that are a straight
        line. Original version:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats
        .zscore.html

        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over
            the whole array a.

        ddof : int, optional
            Degrees of freedom correction in the calculation of the standard
            deviation. Default is 0.

        Returns
        -------
        zscore : array_like
            The z-scores, standardized by mean and standard deviation of
            input array a.
        """
        zscored = np.empty(a.shape)
        for i, j in enumerate(a):
            # j = np.asanyarray(j)
            sstd = j.std(axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no
            # variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored[i] = np.zeros(len(j))
            else:
                mns = j.mean(axis=axis)
                if axis and mns.ndim < j.ndim:
                    zscored[i] = (j - np.expand_dims(mns, axis=axis)) / np.expand_dims(
                        sstd, axis=axis
                    )
                else:
                    zscored[i] = (j - mns) / sstd
        return zscored
    
    @staticmethod
    def secure_dot_product(w, v):
        """ Compute (w^T)v by two parties
        
        """

        # Bob computes 
        w = w.reshape(-1, 1)
        v = v.reshape(-1, 1)
        if len(w) != len(v):
            raise ValueError('Unequal size, w is of size %d while v is %d' % len(w), len(v))
        d = w.shape[0]
        s = 2
        Q = np.random.random((s, s))
        r = np.random.randint(1, s + 1)
        w_prime = np.concatenate((w, np.ones((2,1))), axis=0)
        X = np.random.random((s, d + 2))
        X[r - 1, :] = np.transpose(w_prime)

        b = np.sum(Q[:, r - 1])
        c = np.zeros((1, d + 2))
        for i in range(s):
            if i != r - 1:
                c += X[i, :] * np.sum(Q[:, i])

        f = np.random.random((d + 2, 1))
        R = np.random.random(3)

        QX = np.matmul(Q, X)
        c_prime = c + np.transpose(f) * R[0] * R[1]
        g = f * R[0] * R[2]

        if glb.has_global and glb.get_value('communication') is not None:
            glb.set_value('communication', glb.get_value('communication') + QX.size + c_prime.size + g.size)

        # Alice computes
        alpha = np.random.random((2, 1))
        v_prime = np.concatenate((v, alpha), axis=0)
        y = np.matmul(QX, v_prime)
        z = np.sum(y)

        a = z - np.matmul(c_prime, v_prime)
        h = np.matmul(np.transpose(g), v_prime)

        if glb.has_global and glb.get_value('communication') is not None:
            glb.set_value('communication', glb.get_value('communication') + a.size + h.size)


        # Bob computes
        beta = (a + h * R[1] / R[2]) / b

        return alpha[:,0], beta[0, 0]
    
    @staticmethod
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
            

class Shapelet:
    """A simple class to model a Shapelet with associated information

    Parameters
    ----------
    series_id: int
        The index of the series within the data (X) that was passed to fit.
    start_pos: int
        The starting position of the shapelet within the original series
    length: int
        The length of the shapelet
    quality: flaot
        The calculated quality measure(here is F-statistic) of this shapelet
    data: array-like
        The (z-normalised) data of this shapelet
    """

    def __init__(self, series_id, start_pos, length, quality, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.quality = quality
        self.data = data

    def __str__(self):
        return "Series ID: {0}, start_pos: {1}, length: {2}, F-stat: {3}," \
               " ".format(self.series_id, self.start_pos,
                          self.length, self.quality)


class ShapeletPQ:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, shapelet):
        heapq.heappush(self._queue,
                       (shapelet.quality, self._index, shapelet))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        return self._queue[0]

    def get_size(self):
        return len(self._queue)

    def get_array(self):
        return self._queue


class ContractedFederatedShapeletTransform(FederatedShapeletTransform):
    """ Contracted shapelet transform
    
    Parameters
    ------------------------------
    min_shapelet_length                 : int, lower bound on candidate
    shapelet lengths (default = 3)
    max_shapelet_length                 : int, upper bound on candidate
    shapelet lengths (default = inf or series length)
    max_shapelets_to_store_per_class    : int, upper bound on number of
    shapelets to retain                    (default = 200 * num_classes)
    time_contract_in_mins                  : float, the number of minutes
    allowed for shapelet extraction (default = 60)
    num_candidates_to_sample_per_case   : int, number of candidate shapelets
    to assess per training series before moving on to
                                          the next series (default = 20)
    random_state                        : RandomState, int, or none: to
    control random state objects for deterministic results (default = None)
    verbose                             : int, level of output printed to
    the console (for information only) (default = 0)
    remove_self_similar                 : boolean, remove overlapping
    "self-similar" shapelets from the final transform (default = True)
    
    Attributes
    ----------

    """
    
    def __init__(
            self,
            min_shapelet_length=3,
            max_shapelet_length=np.inf,
            max_shapelets_to_store=200,
            time_contract_in_mins=60,
            num_candidates_to_sample_per_case=20,
            random_state=None,
            verbose=0,
            remove_self_similar=True
    ):
        self.num_candidates_to_sample_per_case = \
            num_candidates_to_sample_per_case
        self.time_contract_in_mins = time_contract_in_mins

        self.predefined_F_stat_rejection_level = 0
        self.shapelets = None

        super().__init__(min_shapelet_length, max_shapelet_length,
                         max_shapelets_to_store, random_state,
                         verbose, remove_self_similar)