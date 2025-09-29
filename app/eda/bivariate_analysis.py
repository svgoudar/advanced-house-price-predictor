import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.core import BaseAnalyzer
from app.data_injestion import train_df


class BivariateAnalyzer(BaseAnalyzer):
    """
    Bivariate Analysis for House Price Prediction Dataset.
    Analyzes relationships between features and target variable.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.numerical_features = self._get_numerical_features()
        self.categorical_features = self._get_categorical_features()
        self.target = "SalePrice"

    def _get_numerical_features(self) -> list:
        """Get list of numerical features excluding 'Id' and target."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in ["Id", "SalePrice"]:
            if col in numerical_cols:
                numerical_cols.remove(col)
        return numerical_cols

    def _get_categorical_features(self) -> list:
        """Get list of categorical features."""
        return self.df.select_dtypes(include=["object"]).columns.tolist()

    def analyze_numerical_vs_target(self, features=None, figsize=(20, 15)):
        """Analyze relationship between numerical features and target variable."""
        if features is None:
            # Select top correlated features
            correlations = (
                self.df[self.numerical_features + [self.target]]
                .corr()[self.target]
                .drop(self.target)
            )
            features = (
                correlations.abs().sort_values(ascending=False).head(12).index.tolist()
            )

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            f"Numerical Features vs {self.target}", fontsize=16, fontweight="bold"
        )
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, feature in enumerate(features):
            if i >= len(axes):
                break
            # Remove missing values
            mask = ~(self.df[feature].isnull() | self.df[self.target].isnull())
            x = self.df.loc[mask, feature]
            y = self.df.loc[mask, self.target]

            if len(x) > 0:
                # Scatter plot
                axes[i].scatter(x, y, alpha=0.6, s=20)
                # Regression line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes[i].plot(x, p(x), "r--", alpha=0.8)
                correlation = x.corr(y)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(self.target)
                axes[i].set_title(
                    f"{feature} vs {self.target}\nCorr: {correlation:.3f}"
                )
                axes[i].grid(alpha=0.3)

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print statistical analysis
        print("\n" + "=" * 50)
        print(f"NUMERICAL FEATURES vs {self.target.upper()} ANALYSIS")
        print("=" * 50)
        for feature in features:
            mask = ~(self.df[feature].isnull() | self.df[self.target].isnull())
            x = self.df.loc[mask, feature]
            y = self.df.loc[mask, self.target]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            correlation = x.corr(y)
            print(f"\n{feature}:")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  R-squared: {r_value**2:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Slope: {slope:.4f}")

    def analyze_categorical_vs_target(self, features=None, figsize=(20, 15)):
        """Analyze relationship between categorical features and target variable."""
        if features is None:
            features = [
                f for f in self.categorical_features if 2 <= self.df[f].nunique() <= 15
            ][:12]

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            f"Categorical Features vs {self.target}", fontsize=16, fontweight="bold"
        )
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        statistical_results = {}

        for i, feature in enumerate(features):
            if i >= len(axes):
                break
            data_for_plot = self.df[[feature, self.target]].dropna()
            if len(data_for_plot) > 0:
                sns.boxplot(data=data_for_plot, x=feature, y=self.target, ax=axes[i])
                axes[i].set_title(f"{feature} vs {self.target}")
                axes[i].tick_params(axis="x", rotation=45)

                # ANOVA test
                groups = [
                    group[self.target].values
                    for name, group in data_for_plot.groupby(feature)
                ]
                if len(groups) > 1 and all(len(group) > 1 for group in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    statistical_results[feature] = {
                        "f_stat": f_stat,
                        "p_value": p_value,
                    }
                    axes[i].text(
                        0.02,
                        0.98,
                        f"p-value: {p_value:.4f}",
                        transform=axes[i].transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        fontsize=8,
                    )

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print statistical results
        print("\n" + "=" * 50)
        print(f"CATEGORICAL FEATURES vs {self.target.upper()} ANALYSIS")
        print("=" * 50)
        for feature in features:
            if feature in statistical_results:
                result = statistical_results[feature]
                print(f"\n{feature}:")
                print(f"  ANOVA F-statistic: {result['f_stat']:.4f}")
                print(f"  P-value: {result['p_value']:.6f}")
                print(f"  Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}")

    def analyze_numerical_correlations(self, top_n=15, figsize=(15, 12)):
        """Analyze correlations between numerical features."""
        correlation_matrix = self.df[self.numerical_features].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Full correlation heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax1,
            cbar_kws={"shrink": 0.8},
        )
        ax1.set_title("Numerical Features Correlation Matrix")

        # Top correlations
        upper_corr = correlation_matrix.where(mask)
        corr_pairs = (
            upper_corr.stack()
            .reset_index()
            .rename(
                columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation"}
            )
            .sort_values("Correlation", key=abs, ascending=False)
        )
        top_corr = corr_pairs.head(top_n)
        colors = ["red" if x < 0 else "green" for x in top_corr["Correlation"]]
        y_labels = [
            f"{row['Feature1']} - {row['Feature2']}" for _, row in top_corr.iterrows()
        ]

        ax2.barh(range(len(top_corr)), top_corr["Correlation"], color=colors)
        ax2.set_yticks(range(len(top_corr)))
        ax2.set_yticklabels(y_labels)
        ax2.set_xlabel("Correlation")
        ax2.set_title(f"Top {top_n} Feature Correlations")
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print top correlations
        print("\n" + "=" * 50)
        print("NUMERICAL FEATURES CORRELATIONS")
        print("=" * 50)
        print(f"\nTop {top_n} Correlations:")
        for _, row in top_corr.iterrows():
            print(f"  {row['Feature1']} - {row['Feature2']}: {row['Correlation']:.4f}")

        high_corr = corr_pairs[corr_pairs["Correlation"].abs() > 0.8]
        if len(high_corr) > 0:
            print("\n⚠️  High Correlations (>0.8) - Potential Multicollinearity:")
            for _, row in high_corr.iterrows():
                print(
                    f"  {row['Feature1']} - {row['Feature2']}: {row['Correlation']:.4f}"
                )

    def __call__(self, analysis_type="all"):
        """Run bivariate analysis."""
        print("\n" + "=" * 60)
        print("BIVARIATE ANALYSIS - HOUSE PRICE PREDICTION")
        print("=" * 60)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")

        if analysis_type in ["all", "numerical_target"]:
            print("\n📊 Analyzing Numerical Features vs Target...")
            self.analyze_numerical_vs_target()

        if analysis_type in ["all", "categorical_target"]:
            print("\n📋 Analyzing Categorical Features vs Target...")
            self.analyze_categorical_vs_target()

        if analysis_type in ["all", "numerical_correlations"]:
            print("\n🔗 Analyzing Numerical Feature Correlations...")
            self.analyze_numerical_correlations()

        print("\n✅ Bivariate Analysis Complete!")


# Instance for easy usage
bivariate_analyzer = BivariateAnalyzer(train_df)

if __name__ == "__main__":
    # Run complete analysis
    bivariate_analyzer()
