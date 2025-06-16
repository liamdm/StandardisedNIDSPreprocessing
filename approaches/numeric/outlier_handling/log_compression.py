from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from approaches.base_approach import BaseApproach

class LogCompression(BaseApproach, BaseEstimator, TransformerMixin):
    """
    Applies logarithmic compression to numerical features.

    Parameters
    ----------
    base : int, default=10
        The logarithmic base to use. Must be greater than 1 and less than or equal to 32.

    force_pandas_mode : bool, default=False
        If True, treats the input as a pandas DataFrame to preserve column names
        and apply per-column transformations.

    nan_value : float or None, default=np.nan
        Replacement value for invalid log operations (e.g. log(0), log of negative numbers):

        - If `np.nan` (default), invalid values are replaced with NaN.
        - If a float (e.g. 0.0 or -1.0), invalid values are replaced with that value.
        - If `-np.inf`, retains NumPy's default behavior.

    Notes
    -----
    - Applies `log(x) / log(base)` transformation element-wise.
    - Values less than or equal to 0 are considered invalid for log and are handled as specified by `nan_value`.
    - Only numeric data is supported. Non-numeric columns must be handled externally.

    Tuneable parameters (others are excluded from get/set params):
    - base
    """
    def __init__(self, *, base: int = 10, force_pandas_mode: bool = False, nan_value: float = np.nan):
        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=[
            "base"
        ])

        assert 2 <= base <= 32, "Log base must be between 2 and 32."

        self.base: int                  = base
        self.nan_value: float           = nan_value

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        X_logged             = np.asarray(X).astype("float32")
        invalid_mask         = X_logged <= 0
        X_logged[invalid_mask] = 100.0  # dummy value to avoid log error
        X_logged             = np.log(X_logged) / np.log(self.base)
        X_logged[invalid_mask] = self.nan_value

        if isinstance(X, pd.DataFrame):
            X_logged = pd.DataFrame(X_logged, columns=X.columns, index=X.index)
        return X_logged


if __name__ == '__main__':
    # Sample DataFrame with multiple numeric columns
    df = pd.DataFrame({
        "feature_1": [1, 10, 100, 1000],
        "feature_2": [0.1, 1, 10, 100],
        "feature_3": [0, -5, 20, 50]  # Contains 0 and a negative to test clipping
    })

    print("Original DataFrame:")
    print(df)

    # Instantiate and apply the transformer
    log_transformer = LogCompression(base=10, force_pandas_mode=True)
    df_transformed = log_transformer.fit_transform(df)

    print("\nLog-Compressed DataFrame:")
    print(df_transformed)