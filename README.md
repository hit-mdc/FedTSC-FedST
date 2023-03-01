# FedTSC-FedST
Souce code of the paper "FedST: Secure Federated Shapelet Transformation for Interpretable Time Series Classification". 

### Import
```
from FedST import ContractedFederatedShapeletTransform
```

### Initialization
```
fedst = ContractedFederatedShapeletTransform(time_contract_in_mins=60)
```

### Federated training and transformation
```
fedst.fit(X, y)
X_transform = fedst.transform(X)
```

### Local transformation
```
X_transform_locally = fedst.transform_locally(X)
```

