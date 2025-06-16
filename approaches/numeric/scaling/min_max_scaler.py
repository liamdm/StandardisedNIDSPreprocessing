from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from approaches.base_approach import BaseApproach


class MinMaxScaler(BaseApproach, BaseEstimator, TransformerMixin):
    """
    Applies Min-Max scaling to each feature, rescaling values to a given output range.

    Parameters
    ----------
    output_range : float or tuple of two floats, default=(0.0, 1.0)
        The desired range of transformed data. If a single float is provided, both min and max
        are set to that value. If a tuple is provided, it should contain (min, max).
    force_pandas_mode : bool, default=False
        If True, preserves pandas DataFrame structure (columns/index).

    Notes
    -----
    - Rescales each feature independently to the specified output range.
    - Sensitive to outliers.

    Example
    -------
    >>> scaler = MinMaxScaler(output_range=(0, 1), force_pandas_mode=True)
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    >>> scaler.fit_transform(df)
    """

    def __init__(self, output_range: Union[float, Tuple[float, float]] = (0.0, 1.0), *, force_pandas_mode: bool = False):
        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=["output_range_min", "output_range_max"])

        if isinstance(output_range, tuple):
            self.output_range_min = output_range[0] if len(output_range) > 0 else 0.0
            self.output_range_max = output_range[1] if len(output_range) > 1 else 1.0
        else:
            self.output_range_min = self.output_range_max = output_range

        # Learned during fit
        self.min_: Optional[float]      = None
        self.max_: Optional[float]      = None
        self.range_: Optional[float]    = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X).astype("float32")
        self.min_ = X_arr.min(axis=0)
        self.max_ = X_arr.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        X_input = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        X_arr = np.asarray(X_input).astype("float32")
        X_scaled = (X_arr - self.min_) / self.range_
        X_scaled = X_scaled * (self.output_range_max - self.output_range_min) + self.output_range_min

        if isinstance(X_input, pd.DataFrame):
            return pd.DataFrame(X_scaled, columns=X_input.columns, index=X_input.index)

        return X_scaled


if __name__ == '__main__':
    df = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [10, 20, 30]
    })

    scaler = MinMaxScaler(output_range=(0, 1), force_pandas_mode=True)
    print(scaler.fit_transform(df))
