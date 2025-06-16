from typing import Optional, List, Union, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from approaches.base_approach import BaseApproach

class TopNOneHotEncoder(BaseApproach, BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoder supporting both pandas and numpy input.

    Parameters
    ----------
    drop_first : bool, default=True
        If True, drops the first category in each feature to avoid multicollinearity.

    n_most_frequent : Optional[int], default=None
        If > 0, encodes only the top n most frequent values, determined based on statistics from the training set
        Otherwise, encodes all levels

    force_pandas_mode : bool, default=False
        If True, preserves pandas DataFrame structure (columns/index).

    Notes
    -----
    - Works with both pandas DataFrames and numpy arrays.
    - Learns categories during `fit`, applies consistent encoding on `transform`.

    Example
    -------
    >>> encoder = TopNOneHotEncoder(drop='first', force_pandas_mode=True)
    >>> df = pd.DataFrame({'color': ['red', 'green', 'blue']})
    >>> encoder.fit_transform(df)
    """

    def __init__(self, *,  n_most_frequent:Optional[int] = None, drop_first: bool = True, in_place:bool=True, force_pandas_mode: bool = False):
        assert n_most_frequent is None or n_most_frequent > 0, "n_most_frequent must be None, or greater than 0"

        super().__init__(force_pandas_mode=force_pandas_mode, tuneable_parameters=["drop_first", "n_most_frequent"])

        self.in_place           = in_place
        self.n_most_frequent    = n_most_frequent
        self.drop_first         = drop_first

        # Learned during fit
        self._categories: Dict[Union[str, int], np.ndarray] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self._categories.clear()

        X = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X

        if isinstance(X, np.ndarray):
            assert len(X.shape) == 2, "When handling numpy arrays, the shape must be (n_rows, n_features)"

        column_labels = list(X.columns) if isinstance(X, pd.DataFrame) else  list(range(len(X)))

        for column_i in range(len(column_labels)):
            column_label    = column_labels[column_i]
            X_column        = X[column_label] if isinstance(X, pd.DataFrame) else X[column_i]

            c_features, c_counts = np.unique(
                X_column, return_counts=True
            )

            feature_order = np.argsort(c_counts).reshape(-1)[::-1]
            if self.n_most_frequent:
                feature_order = feature_order[:self.n_most_frequent]

            self._categories[
                column_label
            ] = c_features[feature_order]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):

        X       = pd.DataFrame(X) if self.force_pandas_mode and not isinstance(X, pd.DataFrame) else X
        is_df   = isinstance(X, pd.DataFrame)

        if not self.in_place:
            X = X.copy() if is_df else np.copy(X)

        if isinstance(X, np.ndarray):
            assert len(X.shape) == 2, "When handling numpy arrays, the shape must be (n_rows, n_features)"

        column_labels   = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(len(X)))

        X_output = X if is_df else []

        for column_i in range(len(column_labels)):
            column_label    = column_labels[column_i]
            X_column        = X[column_label] if isinstance(X, pd.DataFrame) else X[column_i]

            mapping = self._categories[column_label]

            drop_index          = 0 if self.drop_first == 'first' else None
            category_indices    = {cat: idx for idx, cat in enumerate(mapping) if drop_index is None or idx != drop_index}
            valid_categories    = list(category_indices.keys())

            dummies         = pd.get_dummies(X_column)
            dummies         = dummies.reindex(columns=valid_categories, fill_value=0)
            dummies.columns = [f"{column_label}_{cat}" for cat in valid_categories]

            if is_df:
                X_output = pd.concat([X_output.drop(columns=[column_label]), dummies.astype(np.float16)], axis=1)
            else:
                print("!!", X_column.shape, "==>", dummies.shape)
                print()
                X_output.append(dummies)

        if not is_df:
            X_output = np.hstack(dummies)

        return X_output

if __name__ == '__main__':
    encoder = TopNOneHotEncoder(drop_first=False, force_pandas_mode=False)

    # With numpy
    arr = np.array([
        ['red', 'circle'],
        ['green', 'square'],
        ['blue', 'oval'],
        ['green', 'triangle']
    ])

    df = pd.DataFrame({
        'color': ['red', 'green', 'blue', 'green'],
        'shape': ['circle', 'square', 'oval', 'triangle']
    })

    print(df)
    v = encoder.fit_transform(df)
    print(v)
    print(v.values.shape)
    print("=" * 20)

    out_arr = encoder.fit_transform(arr)
    print(arr.shape)
    print(out_arr.shape)
