from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class BaseApproach(BaseEstimator, TransformerMixin):
    def __init__(self, *, force_pandas_mode:bool=False, tuneable_parameters:List[str]=None):
        self.force_pandas_mode = force_pandas_mode
        self.tuneable_parameters: List[str] = tuneable_parameters if tuneable_parameters is not None else []

    def get_params(self, deep=True):
        return dict([
            (k, getattr(self, k)) for k in self.tuneable_parameters
        ])

    def set_params(self, **params):
        for k, v in [(k, v) for (k, v) in params.items() if k in self.tuneable_parameters]:
            setattr(self, k, v)



