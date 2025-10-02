import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scale continuous numeric features in a DataFrame using different scaling strategies.

    This transformer automatically detects continuous numeric columns
    and applies the chosen scaling method. It supports inverse_transform
    for integration in pipelines. It can also visualize before/after scaling.

    Parameters
    ----------
    method : str, default="auto"
        Scaling strategy to apply:
        - "auto": automatically chooses scaler based on skew/outliers.
        - "standard": use StandardScaler (mean=0, std=1).
        - "minmax": use MinMaxScaler (scale to [0,1]).
        - "robust": use RobustScaler (median=0, IQR=1, robust to outliers).

    skew_threshold : float, default=0.5
        Absolute skew above which 'auto' mode may use RobustScaler.

    min_unique : int, default=30
        Minimum number of unique values for a column to be considered continuous.

    visualize_flag : bool, default=True
        Whether to plot boxplots before and after scaling in fit_transform.

    Attributes
    ----------
    continuous_features_ : list
        List of continuous numeric columns detected in the DataFrame.

    scalers_ : dict
        Dictionary mapping each continuous column to its fitted scaler.

    Notes on when to use each scaler
    --------------------------------
    - StandardScaler: Use when features are roughly Gaussian and free of extreme outliers.
    - MinMaxScaler: Use when you want all features in the [0,1] range, e.g., for neural networks.
    - RobustScaler: Use when features contain outliers or are highly skewed.
    - Auto mode: Automatically applies RobustScaler for skewed or repeated-value features,
      otherwise StandardScaler.
    """

    def __init__(
        self,
        method="auto",
        target=None,
        skew_threshold=0.5,
        min_unique=30,
        visualize=True,
    ):
        self.target = target
        self.method = method
        self.skew_threshold = skew_threshold
        self.min_unique = min_unique
        self.visualize = visualize
        self.continuous_features_ = []
        self.scalers_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.continuous_features_ = [
            col for col in numeric_features if X[col].nunique() > self.min_unique
        ]
        if self.target and self.target in self.continuous_features_:
            self.continuous_features_.remove(self.target)
        for col in self.continuous_features_:
            skewness = X[col].skew()

            if self.method == "standard":
                scaler = StandardScaler().fit(X[[col]])
            elif self.method == "minmax":
                scaler = MinMaxScaler().fit(X[[col]])
            elif self.method == "robust":
                scaler = RobustScaler().fit(X[[col]])
            else:  # auto
                if (
                    abs(skewness) > self.skew_threshold
                    or X[col].nunique() / X.shape[0] < 0.05
                ):
                    scaler = RobustScaler().fit(X[[col]])
                else:
                    scaler = StandardScaler().fit(X[[col]])

            self.scalers_[col] = scaler
        return self

    def transform(self, X):
        X_new = X.copy()
        for col, scaler in self.scalers_.items():
            X_new[col] = scaler.transform(X_new[[col]])
        if self.visualize:
            self.visualize_distribution(X, X_new)
        return X_new

    def inverse_transform(self, X):
        X_orig = X.copy()
        for col, scaler in self.scalers_.items():
            X_orig[col] = scaler.inverse_transform(X_orig[[col]])
        return X_orig

    def visualize_distribution(self, X, X_scaled=None):
        """Visualize boxplots before and after scaling for continuous features."""
        if len(self.continuous_features_) == 0:
            print("No continuous features detected.")
            return

        X_scaled = X_scaled if X_scaled is not None else self.transform(X)
        n_features = len(self.continuous_features_)

        fig, axes = plt.subplots(n_features, 2, figsize=(12, 4 * n_features))
        if n_features == 1:
            axes = np.expand_dims(axes, axis=0)  # Make 2D array for consistent indexing

        for i, col in enumerate(self.continuous_features_):
            sns.boxplot(x=X[col], ax=axes[i, 0], color="skyblue")
            axes[i, 0].set_title(f"Original: {col}")

            sns.boxplot(x=X_scaled[col], ax=axes[i, 1], color="lightgreen")
            axes[i, 1].set_title(f"Scaled: {col}")

        plt.tight_layout()
        plt.show()
