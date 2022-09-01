import numpy as np
from sklearn.utils import check_random_state

def federated_data_split(data, *, num_parties=3, ratio=None, random_state=None):
    """ Split input sktime formatted time series data into federated parties.
    
    Parameters:
    ----------------------------------
    data:                    Input data, 2D array or dataframe
    num_parties:             The number of federated parties
    ratio:                   List, the ratio of data in each party.
                             Default = None, split the data evenly.
    random_state:            The random seed
    
    Return:
    ----------------------------------
    T_fed:                  List of dataframe, federated training data
    y_fed:                  List of ndarray, federated targets
    """
    
    
    if ratio is None:
        ratio = [1 / num_parties] * num_parties
    else:
        ratio = [r / sum(ratio) for r in ratio]
    
    cum_ratio = np.cumsum([0] + ratio)
    
    rst = check_random_state(random_state)
    
    T = data.iloc[:,0].to_frame()
    y = data.iloc[:,1].to_numpy()
    
    classes = np.unique(data.iloc[:, 1].to_numpy())
    n_classes = len(classes)

    data_idx_per_class = []
    data_splits_per_class = []
    for c in classes:
        data_idx = np.where(y == c)[0]
        rst.shuffle(data_idx)
        data_idx_per_class.append(data_idx)
        data_splits_per_class.append(np.int32(np.round(cum_ratio * len(data_idx))))
    
    T_fed = []
    y_fed = []
    
    for i in range(num_parties):
        
        idx = np.concatenate(
            [data_idx_per_class[c][data_splits_per_class[c][i]:data_splits_per_class[c][i+1]]\
             for c in range(n_classes)]
        )
        T_fed.append(T.iloc[idx])
        y_fed.append(y[idx])
    
    return T_fed, y_fed