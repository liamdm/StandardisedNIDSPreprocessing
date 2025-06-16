from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np
from typing import Union, Optional

from approaches.base_approach import BaseApproach


class QuantileClipping(BaseApproach, BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that clips feature values based on quantile thresholds.

    This transformer computes the lower and upper quantiles of each feature during `fit`,
    and during `transform` it clips values below the lower quantile and above the upper quantile
    to the respective quantile values. This is useful for mitigating the impact of outliers
    in tabular data.

    Parameters
    ----------
    lower_quantile : float, default=0.01
        The lower quantile (between 0 and 1) to use for clipping. Values below this
        quantile will be clipped to this threshold.

    upper_quantile : float, default=0.99
        The upper quantile (between 0 and 1) to use for clipping. Values above this
        quantile will be clipped to this threshold.

    force_pandas_mode : bool, default=False
        If True, forces the input to be treated as a pandas DataFrame for quantile computation
        and clipping. This enables per-column clipping and supports pandas-specific features.

    Attributes
    ----------
    lower_bounds_ : pandas.Series or numpy.ndarray
        The computed lower quantile thresholds for each feature after fitting.

    upper_bounds_ : pandas.Series or numpy.ndarray
        The computed upper quantile thresholds for each feature after fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> X = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 10000]])
    >>> qc = QuantileClipping(lower_quantile=0.1, upper_quantile=0.9)
    >>> qc.fit_transform(X)
    array([[1, 100],
           [2, 200],
           [3, 300],
           [4, 400],
           [5, 400]])  # last row clipped at 90th percentile of column 2

    Notes
    -----
    - If `force_pandas_mode` is True and the input is not a DataFrame, it will be
      converted to one. Otherwise, NumPy-based clipping will be used.
    - When using NumPy mode, clipping is done across axis -1 (per feature).

    Tuneable parameters (others are excluded from get/set params):
    - lower_quantile
    - upper_quantile
    """

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99, *, force_pandas_mode:bool = False):
        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=[
            "lower_quantile", "upper_quantile"
        ])

        self.lower_quantile: float = lower_quantile
        self.upper_quantile: float = upper_quantile

        # Learned during fit
        self.lower_bounds_: Optional[float] = None
        self.upper_bounds_: Optional[float] = None

    def fit(self, X, y=None, **fit_params):
        # Store quantile thresholds
        X = pd.DataFrame(X) if self.force_pandas_mode else X

        if isinstance(X, pd.DataFrame):
            self.lower_bounds_ = X.quantile(self.lower_quantile)
            self.upper_bounds_ = X.quantile(self.upper_quantile)
        else:
            X = X if isinstance(X, np.ndarray) else np.ndarray(X)
            self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=-1)
            self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=-1)

        return self

    def transform(self, X):
        X = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        else:
            X = X if isinstance(X, np.ndarray) else np.ndarray(X)
            return np.clip(X, self.lower_bounds_, self.upper_bounds_)