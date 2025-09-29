import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

from app.utils.core import BaseAnalyzer
from app.data_injestion import train_df


class UnivariateAnalyzer(BaseAnalyzer):
    """
    Univariate Analysis for House Price Prediction Dataset
    Analyzes individual features (both categorical and numerical)
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.numerical_features = self._get_numerical_features()
        self.categorical_features = self._get_categorical_features()

    def _get_numerical_features(self) -> list:
        """Get list of numerical features excluding Id and target"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove Id and target variable
        if "Id" in numerical_cols:
            numerical_cols.remove("Id")
        if "SalePrice" in numerical_cols:
            numerical_cols.remove("SalePrice")
        return numerical_cols

    def _get_categorical_features(self) -> list:
        """Get list of categorical features"""
        return self.df.select_dtypes(include=["object"]).columns.tolist()

    def analyze_target_variable(self, figsize=(15, 10)):
        """Comprehensive analysis of the target variable (SalePrice)"""
        if "SalePrice" not in self.df.columns:
            print("Target variable 'SalePrice' not found in dataset")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            "Target Variable Analysis: SalePrice", fontsize=16, fontweight="bold"
        )

        # Distribution plot
        sns.histplot(data=self.df, x="SalePrice", kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Distribution of SalePrice")
        axes[0, 0].set_xlabel("Sale Price ($)")

        # Box plot
        sns.boxplot(data=self.df, y="SalePrice", ax=axes[0, 1])
        axes[0, 1].set_title("Box Plot of SalePrice")
        axes[0, 1].set_ylabel("Sale Price ($)")

        # Q-Q plot
        stats.probplot(self.df["SalePrice"], dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title("Q-Q Plot (Normal Distribution)")

        # Log transformation
        log_saleprice = np.log1p(self.df["SalePrice"])
        sns.histplot(log_saleprice, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Log-transformed SalePrice")
        axes[1, 0].set_xlabel("Log(Sale Price)")

        # Log Q-Q plot
        stats.probplot(log_saleprice, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot (Log-transformed)")

        # Statistics
        stats_text = f"""
        Statistics:
        Mean: ${self.df['SalePrice'].mean():,.0f}
        Median: ${self.df['SalePrice'].median():,.0f}
        Std: ${self.df['SalePrice'].std():,.0f}
        Skewness: {self.df['SalePrice'].skew():.3f}
        Kurtosis: {self.df['SalePrice'].kurtosis():.3f}
        Min: ${self.df['SalePrice'].min():,.0f}
        Max: ${self.df['SalePrice'].max():,.0f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center")
        axes[1, 2].axis("off")
        axes[1, 2].set_title("Summary Statistics")

        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print("\n" + "=" * 50)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 50)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nSalePrice Statistics:")
        print(self.df["SalePrice"].describe())
        print(f"\nSkewness: {self.df['SalePrice'].skew():.4f}")
        print(f"Kurtosis: {self.df['SalePrice'].kurtosis():.4f}")

    def analyze_numerical_features(self, features=None, figsize=(20, 15)):
        """Analyze numerical features with distributions and statistics"""
        if features is None:
            features = self.numerical_features[
                :20
            ]  # Limit to first 20 for visualization

        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            "Numerical Features Distribution Analysis", fontsize=16, fontweight="bold"
        )

        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, feature in enumerate(features):
            if i >= len(axes):
                break

            # Handle missing values
            data = self.df[feature].dropna()

            if len(data) > 0:
                sns.histplot(data, kde=True, ax=axes[i])
                axes[i].set_title(
                    f"{feature}\n(n={len(data)}, missing={self.df[feature].isnull().sum()})"
                )
                axes[i].tick_params(axis="x", rotation=45)

                # Add statistics as text
                stats_text = f"Mean: {data.mean():.1f}\nStd: {data.std():.1f}\nSkew: {data.skew():.2f}"
                axes[i].text(
                    0.7,
                    0.8,
                    stats_text,
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=8,
                )

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n" + "=" * 50)
        print("NUMERICAL FEATURES SUMMARY")
        print("=" * 50)
        print(f"Total numerical features: {len(self.numerical_features)}")
        print(f"Features analyzed: {len(features)}")

        # Missing values summary
        missing_summary = self.df[features].isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\nFeatures with missing values:")
            for feature, missing_count in missing_summary[missing_summary > 0].items():
                missing_pct = (missing_count / len(self.df)) * 100
                print(f"  {feature}: {missing_count} ({missing_pct:.1f}%)")

        # Statistical summary
        print(f"\nStatistical Summary:")
        print(self.df[features].describe())

    def analyze_categorical_features(self, features=None, figsize=(20, 15)):
        """Analyze categorical features with value counts and distributions"""
        if features is None:
            features = self.categorical_features[:20]  # Limit to first 20

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            "Categorical Features Distribution Analysis", fontsize=16, fontweight="bold"
        )

        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, feature in enumerate(features):
            if i >= len(axes):
                break

            # Get value counts
            value_counts = self.df[feature].value_counts()

            if len(value_counts) > 0:
                # Limit categories for visualization
                if len(value_counts) > 15:
                    top_categories = value_counts.head(15)
                    axes[i].bar(range(len(top_categories)), top_categories.values)
                    axes[i].set_xticks(range(len(top_categories)))
                    axes[i].set_xticklabels(
                        top_categories.index, rotation=45, ha="right"
                    )
                    axes[i].set_title(
                        f"{feature} (Top 15)\nTotal categories: {len(value_counts)}"
                    )
                else:
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha="right")
                    axes[i].set_title(f"{feature}\nCategories: {len(value_counts)}")

                axes[i].set_ylabel("Count")

                # Add missing values info
                missing_count = self.df[feature].isnull().sum()
                if missing_count > 0:
                    axes[i].text(
                        0.7,
                        0.9,
                        f"Missing: {missing_count}",
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
                        fontsize=8,
                    )

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print detailed categorical analysis
        print("\n" + "=" * 50)
        print("CATEGORICAL FEATURES SUMMARY")
        print("=" * 50)
        print(f"Total categorical features: {len(self.categorical_features)}")
        print(f"Features analyzed: {len(features)}")

        for feature in features:
            print(f"\n{feature}:")
            value_counts = self.df[feature].value_counts()
            missing_count = self.df[feature].isnull().sum()

            print(f"  Unique values: {len(value_counts)}")
            print(
                f"  Missing values: {missing_count} ({(missing_count/len(self.df)*100):.1f}%)"
            )
            print(f"  Top 5 categories:")
            for cat, count in value_counts.head().items():
                pct = (count / len(self.df)) * 100
                print(f"    {cat}: {count} ({pct:.1f}%)")

    def analyze_feature_correlations_with_target(self, target="SalePrice", top_n=20):
        """Analyze correlation of numerical features with target variable"""
        if target not in self.df.columns:
            print(f"Target variable '{target}' not found in dataset")
            return

        # Calculate correlations
        correlations = (
            self.df[self.numerical_features + [target]].corr()[target].drop(target)
        )
        correlations = correlations.sort_values(key=abs, ascending=False)

        # Plot top correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Top positive and negative correlations
        top_corr = correlations.head(top_n)
        colors = ["red" if x < 0 else "green" for x in top_corr.values]

        ax1.barh(range(len(top_corr)), top_corr.values, color=colors)
        ax1.set_yticks(range(len(top_corr)))
        ax1.set_yticklabels(top_corr.index)
        ax1.set_xlabel("Correlation with SalePrice")
        ax1.set_title(f"Top {top_n} Feature Correlations with {target}")
        ax1.grid(axis="x", alpha=0.3)

        # Correlation heatmap for top features
        top_features = top_corr.head(15).index.tolist() + [target]
        corr_matrix = self.df[top_features].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax2,
            cbar_kws={"shrink": 0.8},
        )
        ax2.set_title(f"Correlation Heatmap: Top Features with {target}")

        plt.tight_layout()
        plt.show()

        # Print correlation summary
        print("\n" + "=" * 50)
        print(f"FEATURE CORRELATIONS WITH {target.upper()}")
        print("=" * 50)
        print(f"\nTop 10 Positive Correlations:")
        positive_corr = correlations[correlations > 0].head(10)
        for feature, corr in positive_corr.items():
            print(f"  {feature}: {corr:.4f}")

        print(f"\nTop 10 Negative Correlations:")
        negative_corr = correlations[correlations < 0].tail(10)
        for feature, corr in negative_corr.items():
            print(f"  {feature}: {corr:.4f}")

    def generate_missing_values_report(self):
        """Generate comprehensive missing values report"""
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) == 0:
            print("No missing values found in the dataset!")
            return

        missing_percent = (missing_data / len(self.df)) * 100

        # Create missing values plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Bar plot
        ax1.bar(range(len(missing_data)), missing_data.values)
        ax1.set_xticks(range(len(missing_data)))
        ax1.set_xticklabels(missing_data.index, rotation=45, ha="right")
        ax1.set_ylabel("Number of Missing Values")
        ax1.set_title("Missing Values Count by Feature")
        ax1.grid(axis="y", alpha=0.3)

        # Percentage plot
        ax2.bar(range(len(missing_percent)), missing_percent.values, color="orange")
        ax2.set_xticks(range(len(missing_percent)))
        ax2.set_xticklabels(missing_percent.index, rotation=45, ha="right")
        ax2.set_ylabel("Percentage of Missing Values")
        ax2.set_title("Missing Values Percentage by Feature")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print missing values report
        print("\n" + "=" * 50)
        print("MISSING VALUES REPORT")
        print("=" * 50)
        print(f"Total features with missing values: {len(missing_data)}")
        print(f"Total missing values: {missing_data.sum()}")
        print(
            f"Overall missing percentage: {(missing_data.sum() / (len(self.df) * len(self.df.columns))) * 100:.2f}%"
        )

        print(f"\nDetailed Missing Values Report:")
        for feature in missing_data.index:
            count = missing_data[feature]
            pct = missing_percent[feature]
            feature_type = (
                "Categorical" if feature in self.categorical_features else "Numerical"
            )
            print(f"  {feature} ({feature_type}): {count} values ({pct:.2f}%)")

    def __call__(self, analysis_type="all"):
        """Main method to run univariate analysis"""
        print("\n" + "=" * 60)
        print("UNIVARIATE ANALYSIS - HOUSE PRICE PREDICTION")
        print("=" * 60)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")

        if analysis_type in ["all", "target"]:
            print("\n🎯 Analyzing Target Variable...")
            self.analyze_target_variable()

        if analysis_type in ["all", "numerical"]:
            print("\n📊 Analyzing Numerical Features...")
            self.analyze_numerical_features()

        if analysis_type in ["all", "categorical"]:
            print("\n📋 Analyzing Categorical Features...")
            self.analyze_categorical_features()

        if analysis_type in ["all", "correlations"]:
            print("\n🔗 Analyzing Feature Correlations...")
            self.analyze_feature_correlations_with_target()

        if analysis_type in ["all", "missing"]:
            print("\n❓ Analyzing Missing Values...")
            self.generate_missing_values_report()

        print("\n✅ Univariate Analysis Complete!")


# Create instance for easy usage
univariate_analyzer = UnivariateAnalyzer(train_df)

if __name__ == "__main__":
    # Run complete analysis
    univariate_analyzer()
