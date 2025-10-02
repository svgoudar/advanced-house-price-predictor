import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Continuous Feature Transformer for regression or general numeric datasets.

    Automatically detects continuous numerical features (by minimum unique values),
    calculates skewness, and applies appropriate transformations to reduce skew:

    - log1p: for high positive skew and strictly positive values
    - Box-Cox: for moderate positive skew and strictly positive values
    - Yeo-Johnson: for negative skew or features containing zeros/negative values
    - None: for features with low skewness

    Can visualize the original vs transformed distributions using seaborn.

    Parameters
    ----------
    skew_threshold : float, default=0.5
        Minimum absolute skewness required to apply any transformation.
    # log_skew_threshold : float, default=1.0
        # Skew above this threshold uses log1p transformation (if all values > 0).
    min_unique : int, default=30
        Minimum distinct values to consider a feature continuous.
    visualize : bool, default=True
        If True, will visualize original vs transformed distributions after fit_transform.

    Attributes
    ----------
    continuous_features_ : list
        List of features identified as continuous based on min_unique.
    transformers_ : dict
        Mapping of feature name -> (transformation method, fitted transformer object)
    """

    def __init__(self, skew_threshold=0.5, min_unique=30, visualize=True):
        self.skew_threshold = skew_threshold
        self.min_unique = min_unique
        self.visualize = visualize
        self.continuous_features_ = []
        self.transformers_ = {}

    def fit(self, X, y=None):
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.continuous_features_ = [
            col
            for col in numeric_features
            if X[col].nunique() > self.min_unique and not col.endswith("_encoded")
        ]
        for col in self.continuous_features_:
            skewness = X[col].skew()
            col_min = X[col].min()

            if abs(skewness) > self.skew_threshold:
                if skewness > 0.5 and col_min > 0:
                    self.transformers_[col] = ("log1p", None)
                elif 0 < skewness <= 0.5 and col_min > 0:
                    pt = PowerTransformer(method="box-cox").fit(X[[col]])
                    self.transformers_[col] = ("box-cox", pt)
                else:
                    pt = PowerTransformer(method="yeo-johnson").fit(X[[col]])
                    self.transformers_[col] = ("yeo-johnson", pt)
            else:
                self.transformers_[col] = ("none", None)
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        for col, (method, transformer) in self.transformers_.items():
            if method == "log1p":
                X_new[col] = np.log1p(X_new[col])
            elif method in ["box-cox", "yeo-johnson"]:
                X_new[col] = transformer.transform(X_new[[col]])
        if self.visualize:
            self.visualize_distributions(X, X_new)
        return X_new

    def inverse_transform(self, X):
        X_orig = X.copy()
        for col, (method, transformer) in self.transformers_.items():
            if method == "log1p":
                X_orig[col] = np.expm1(X_orig[col])
            elif method in ["box-cox", "yeo-johnson"]:
                X_orig[col] = transformer.inverse_transform(X_orig[[col]])
        return X_orig

    def visualize_distributions(self, X, X_transformed):
        """Visualize all continuous features before and after transformation using seaborn."""
        n_features = len(self.continuous_features_)
        if n_features == 0:
            print("No continuous features detected for visualization.")
            return

        n_cols = 2  # before and after
        n_rows = n_features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)  # make it 2D for consistency

        for i, col in enumerate(self.continuous_features_):
            sns.histplot(X[col], bins=30, kde=True, color="skyblue", ax=axes[i, 0])
            axes[i, 0].set_title(f"Original: {col}\nSkew={X[col].skew():.2f}")

            sns.histplot(
                X_transformed[col], bins=30, kde=True, color="lightgreen", ax=axes[i, 1]
            )
            axes[i, 1].set_title(
                f"Transformed: {col}\nSkew={X_transformed[col].skew():.2f}"
            )

        plt.tight_layout()
        plt.show()
