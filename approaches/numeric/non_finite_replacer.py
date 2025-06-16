import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NonFiniteReplacer(BaseEstimator, TransformerMixin):
    """
    Replaces non-finite values (NaN, +inf, -inf) in numerical data with a specified replacement value.

    Parameters
    ----------
    replacement_value : float, default=0.0
        The value to use in place of any non-finite (NaN, +inf, -inf) values.

    force_pandas_mode : bool, default=False
        If True, treats the input as a pandas DataFrame to preserve column names and index.

    Notes
    -----
    - Applies replacement to all non-finite values (np.isnan(x) or np.isinf(x)).
    - Only numeric data is supported. Non-numeric columns must be handled externally.

    Tuneable parameters (others are excluded from get/set params):
    - replacement_value (not-recommended)
    """

    def __init__(self, replacement_value: float = 0.0, *, force_pandas_mode: bool = False):
        self.replacement_value: float = replacement_value
        self.force_pandas_mode: float = force_pandas_mode

    def get_params(self, deep=True):
        return {"replacement_value": self.replacement_value}

    def set_params(self, **params):
        for key, value in params.items():
            if key in ["replacement_value"]:
                setattr(self, key, value)
        return self

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        X       = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X
        X_arr   = np.asarray(X).astype("float32")

        X_arr[~np.isfinite(X_arr)] = self.replacement_value

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_arr, columns=X.columns, index=X.index)

        return X_arr

if __name__ == '__main__':
    df = pd.DataFrame({
        "a": [1, np.nan, 3, np.inf],
        "b": [-np.inf, 5, 6, np.nan]
    })

    replacer = NonFiniteReplacer(replacement_value=-1, force_pandas_mode=False)
    print(replacer.fit_transform(df))
