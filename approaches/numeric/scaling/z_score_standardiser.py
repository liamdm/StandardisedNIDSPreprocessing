import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from approaches.base_approach import BaseApproach


class ZScoreStandardiser(BaseApproach, BaseEstimator, TransformerMixin):
    """
    Applies Z-score standardisation: (X - mean) / std.

    Parameters
    ----------
    force_pandas_mode : bool, default=False
        If True, preserves pandas DataFrame structure (columns/index).
        If False, np.ndarrays can be passed, or pandas DataFrames, and both will be preserved.

    Notes
    -----
    - Centers data to mean 0 and scales to unit variance.
    - Not robust to outliers.

    Example
    -------
    >>> scaler = ZScoreStandardiser(force_pandas_mode=True)
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    >>> scaler.fit_transform(df)

    Tuneable parameters (others are excluded from get/set params):
    None
    """
    def __init__(self, *, force_pandas_mode: bool = False):
        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=[])

        # Learned during fit
        self.mean_: Optional[float] = None
        self.std_: Optional[float]  = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X).astype("float32")
        self.mean_ = X_arr.mean(axis=0)
        self.std_ = X_arr.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X_input = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        X_arr = np.asarray(X_input).astype("float32")
        X_scaled = (X_arr - self.mean_) / self.std_

        if isinstance(X_input, pd.DataFrame):
            return pd.DataFrame(X_scaled, columns=X_input.columns, index=X_input.index)
        return X_scaled

if __name__ == '__main__':
    df = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [10, 20, 30]
    })

    scaler = ZScoreStandardiser(force_pandas_mode=True)
    print(scaler.fit_transform(df))