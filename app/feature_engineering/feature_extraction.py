import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import f_oneway


# ----------------------------
# Numeric Correlation Selector
# ----------------------------
class NumericCorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, corr_threshold=0.7):
        self.target_col = target_col
        self.corr_threshold = corr_threshold
        self.selected_features_ = []

    def fit(self, X, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"{self.target_col} not found in DataFrame")

        corr = X.corr(numeric_only=True)[self.target_col].dropna()
        self.report_ = corr
        self.selected_features_ = corr[
            (corr.abs() > self.corr_threshold) & (corr.index != self.target_col)
        ].index.tolist()
        return self

    def transform(self, X):
        # Return selected numeric features
        return X


# ----------------------------
# Categorical Correlation Selector
# ----------------------------
class CategoricalCorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target, eta_threshold=0.25, pvalue_threshold=0.05):
        self.target = target
        self.eta_threshold = eta_threshold
        self.pvalue_threshold = pvalue_threshold
        self.selected_features_ = []

    @staticmethod
    def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        mean_total = measurements.mean()
        ss_between = sum(
            len(measurements[fcat == i])
            * (measurements[fcat == i].mean() - mean_total) ** 2
            for i in np.unique(fcat)
        )
        ss_total = sum((measurements - mean_total) ** 2)
        return ss_between / ss_total if ss_total != 0 else 0

    def fit(self, X, y=None):
        results = []

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            groups = [X[self.target][X[col] == lvl] for lvl in X[col].dropna().unique()]
            if len(groups) < 2:
                continue

            f_stat, p_val = f_oneway(*groups)
            eta = self.correlation_ratio(X[col], X[self.target])

            results.append(
                {
                    "Feature": col,
                    "F_stat": f_stat,
                    "p_value": p_val,
                    "Correlation_Ratio": eta,
                }
            )

        df_results = pd.DataFrame(results)
        self.report_ = df_results

        # Filter based on thresholds
        if not df_results.empty:
            mask = (df_results["Correlation_Ratio"] > self.eta_threshold) & (
                df_results["p_value"] < self.pvalue_threshold
            )
            df_results = df_results[mask]
        setattr(self, "selected_features_", df_results["Feature"].tolist())
        return self

    def transform(self, X):
        return X


# ----------------------------
# Correlation Visualizer
# ----------------------------
class CorrelationVisualizer(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target
        self.num_corr_df_ = None
        self.cat_corr_df_ = None

    @staticmethod
    def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        mean_total = measurements.mean()
        ss_between = sum(
            len(measurements[fcat == i])
            * (measurements[fcat == i].mean() - mean_total) ** 2
            for i in np.unique(fcat)
        )
        ss_total = sum((measurements - mean_total) ** 2)
        return ss_between / ss_total if ss_total != 0 else 0

    def fit(self, X, y=None):
        # Numeric correlation

        self.num_corr_df_ = (
            X.corr(method="spearman", numeric_only=True)[self.target]
            .sort_values(ascending=False)
            .to_frame(name=f"Correlation with {self.target}")
        )
        # Categorical correlation
        results = []
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            groups = [X[self.target][X[col] == lvl] for lvl in X[col].dropna().unique()]
            if len(groups) < 2:
                continue
            f_stat, p_val = f_oneway(*groups)
            eta = self.correlation_ratio(X[col], X[self.target])
            results.append(
                {
                    "Feature": col,
                    "F_stat": f_stat,
                    "p_value": p_val,
                    "Correlation_Ratio": eta,
                }
            )
        self.cat_corr_df_ = pd.DataFrame(results).sort_values(
            "Correlation_Ratio", ascending=False
        )
        return self

    def transform(self, X):
        # Just return X to be pipeline compatible
        return X

    def plot(self):
        # Numeric
        plt.figure(figsize=(15, 8))
        plt.subplot(121)
        sns.heatmap(self.num_corr_df_, annot=True, cmap="coolwarm")
        plt.title(f"Numeric Correlation with {self.target}")

        # Categorical
        plt.subplot(122)
        if not self.cat_corr_df_.empty:
            sns.barplot(
                x="Correlation_Ratio",
                y="Feature",
                data=self.cat_corr_df_,
                palette="viridis",
            )
            plt.title(f"Categorical Correlation with {self.target}")
            plt.xlabel("Correlation Ratio (η²)")
            plt.ylabel("Categorical Feature")
        plt.tight_layout()
        plt.show()
