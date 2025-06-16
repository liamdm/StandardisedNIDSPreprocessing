from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from approaches.base_approach import BaseApproach

class RobustScaler(BaseApproach, BaseEstimator, TransformerMixin):
    """
    Applies Robust scaling: (X - median) / IQR.

    Parameters
    ----------
    iqr : int, default=25
        Defines the lower and upper percentile bounds used to compute the interquartile range (IQR).
        Specifically, the lower percentile is `iqr` and the upper percentile is `100 - iqr`.
        Must be in the range (2, 48), which corresponds to valid IQR configurations within the range (2–98) and (48–52).
        For example, with iqr=25, Q1 is the 25th percentile and Q3 is the 75th percentile, resulting in the standard IQR.

    force_pandas_mode : bool, default=False
        If True, preserves pandas DataFrame structure (columns/index).

    Notes
    -----
    - Centers data with the median and scales using the interquartile range (IQR).
    - Robust to outliers due to non-reliance on mean and standard deviation.

    Example
    -------
    >>> scaler = RobustScaler(force_pandas_mode=True)
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 1000]})
    >>> scaler.fit_transform(df)
    """

    def __init__(self, iqr=25, *, force_pandas_mode: bool = False):
        assert 2 < iqr < 48, "IQR range should be betweeen 2 and 98, representing 2-98) and (48-52 respectively"
        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=["iqr"])

        self.iqr = iqr

        # Learned during fit
        self.median_: Optional[float]   = None
        self.q1_: Optional[float]       = None
        self.q3_: Optional[float]       = None
        self.iqr_: Optional[float]      = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X).astype("float32")
        self.median_ = np.median(X_arr, axis=0)
        self.q1_ = np.percentile(X_arr, self.iqr, axis=0)
        self.q3_ = np.percentile(X_arr, 100 - self.iqr, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.iqr_[self.iqr_ == 0] = 1.0
        return self

    def transform(self, X):
        X_input = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        X_arr = np.asarray(X_input).astype("float32")
        X_scaled = (X_arr - self.median_) / self.iqr_

        if isinstance(X_input, pd.DataFrame):
            return pd.DataFrame(X_scaled, columns=X_input.columns, index=X_input.index)

        return X_scaled

if __name__ == '__main__':
    df = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [10, 20, 1000]  # Outlier to show robust effect
    })

    scaler = RobustScaler(force_pandas_mode=True)
    print(scaler.fit_transform(df))
