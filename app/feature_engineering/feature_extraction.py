import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import f_oneway


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Combines numeric and categorical correlation analysis for regression.

    Parameters
    ----------
    target : str
        Name of the target column.
    eta_threshold : float, default=0.25
        Minimum correlation ratio (eta squared) to select categorical features.
    pvalue_threshold : float, default=0.05
        ANOVA p-value threshold to select categorical features.
    visualize : bool, default=False
        If True, plots numeric and categorical correlations.
    """

    def __init__(
        self,
        target=None,
        num_feat_corr_threshold=0.7,
        eta_threshold=0.25,
        pvalue_threshold=0.05,
        visualize=False,
        chosen_features=[],
    ):
        self.target = target
        self.eta_threshold = eta_threshold
        self.num_feat_corr_threshold = num_feat_corr_threshold
        self.pvalue_threshold = pvalue_threshold
        self.visualize = visualize
        self.selected_features_ = []
        self.chosen_features = chosen_features
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
        if self.chosen_features:
            return self
        self.num_corr_df_ = (
            X.corr(method="spearman", numeric_only=True)[self.target]
            .sort_values(ascending=False)
            .to_frame(name=f"Correlation with {self.target}")
        )

        # Extract correlation values as Series for feature selection
        corr_series = X.corr(method="spearman", numeric_only=True)[self.target]
        self.selected_features_ = corr_series[
            (corr_series.abs() > self.num_feat_corr_threshold)
            & (corr_series.index != self.target)
        ].index.tolist()

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

        df_results = pd.DataFrame(results)
        self.cat_corr_df_ = df_results.sort_values("Correlation_Ratio", ascending=False)

        # Feature selection
        if not df_results.empty:
            mask = (df_results["Correlation_Ratio"] > self.eta_threshold) & (
                df_results["p_value"] < self.pvalue_threshold
            )
            filtered = df_results[mask]
            self.selected_features_.extend(filtered["Feature"].tolist())

        if self.visualize:
            self.plot()

        return self

    def transform(self, X):
        if not self.selected_features_ and not self.chosen_features:
            return X
        if self.chosen_features:
            return X[self.chosen_features]
        else:
            cols_to_keep = self.selected_features_

        return X[cols_to_keep + [self.target]]

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
