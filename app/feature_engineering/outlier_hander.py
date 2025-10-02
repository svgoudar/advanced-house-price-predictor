import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and optionally cap/floor outliers in continuous numeric features
    in a DataFrame using the IQR (Interquartile Range) method.

    Parameters
    ----------
    method : str, default="iqr"
        Outlier detection method. Currently only 'iqr' is supported.

    factor : float, default=1.5
        Multiplier for IQR to determine outlier thresholds:
        - Lower bound = Q1 - factor * IQR
        - Upper bound = Q3 + factor * IQR

    min_unique : int, default=30
        Minimum number of unique values for a column to be considered continuous.

    cap : bool, default=True
        If True, replaces outliers beyond bounds with the respective threshold.
        If False, just detects the outliers but does not modify them.

    Attributes
    ----------
    continuous_features_ : list
        List of continuous numeric columns identified in the DataFrame.

    thresholds_ : dict
        Dictionary mapping column names to (lower_bound, upper_bound) tuples.
    """

    def __init__(self, method="iqr", factor=1.5, min_unique=30, cap=True):
        self.method = method
        self.factor = factor
        self.min_unique = min_unique
        self.cap = cap
        self.continuous_features_ = []
        self.thresholds_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.continuous_features_ = [
            col for col in numeric_features if X[col].nunique() > self.min_unique
        ]

        for col in self.continuous_features_:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.thresholds_[col] = (lower_bound, upper_bound)
            else:
                raise NotImplementedError(f"Method '{self.method}' is not implemented.")
        return self

    def transform(self, X):
        X_new = X.copy()
        for col, (lower, upper) in self.thresholds_.items():
            if self.cap:
                X_new[col] = X_new[col].clip(lower=lower, upper=upper)
        return X_new

    def visualize(self, X, bins=30):
        """
        Visualize the distribution of continuous features before and after capping.
        """
        X_transformed = self.transform(X)
        n_features = len(self.continuous_features_)
        if n_features == 0:
            print("No continuous features detected for outlier analysis.")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        n_cols = 2
        n_rows = n_features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, col in enumerate(self.continuous_features_):
            sns.histplot(X[col], bins=bins, kde=True, color="skyblue", ax=axes[i, 0])
            axes[i, 0].set_title(f"Original: {col}")

            sns.histplot(
                X_transformed[col],
                bins=bins,
                kde=True,
                color="lightgreen",
                ax=axes[i, 1],
            )
            axes[i, 1].set_title(f"Capped: {col}")

        plt.tight_layout()
        plt.show()


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        return X[self.columns]
