import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings("ignore")

from app.utils.core import BaseAnalyzer
from app.data_injestion import train_df


class MultivariateAnalyzer(BaseAnalyzer):
    """
    Multivariate Analysis for House Price Prediction Dataset
    Analyzes complex relationships between multiple features simultaneously
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.numerical_features = self._get_numerical_features()
        self.categorical_features = self._get_categorical_features()
        self.target = "SalePrice"
        self.scaler = StandardScaler()

    def _get_numerical_features(self) -> list:
        """Get list of numerical features excluding Id and target"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "Id" in numerical_cols:
            numerical_cols.remove("Id")
        if "SalePrice" in numerical_cols:
            numerical_cols.remove("SalePrice")
        return numerical_cols

    def _get_categorical_features(self) -> list:
        """Get list of categorical features"""
        return self.df.select_dtypes(include=["object"]).columns.tolist()

    def analyze_feature_interactions(self, top_features=10, figsize=(20, 15)):
        """Analyze interactions between top features and their combined effect on target"""
        # Get top correlated features with target
        correlations = (
            self.df[self.numerical_features + [self.target]]
            .corr()[self.target]
            .drop(self.target)
        )
        top_numerical = (
            correlations.abs()
            .sort_values(ascending=False)
            .head(top_features)
            .index.tolist()
        )

        # Create interaction features
        interaction_data = []
        feature_pairs = []

        for i, feat1 in enumerate(top_numerical):
            for j, feat2 in enumerate(top_numerical):
                if i < j:  # Avoid duplicate pairs
                    # Create interaction feature
                    interaction = self.df[feat1] * self.df[feat2]
                    correlation_with_target = interaction.corr(self.df[self.target])

                    if not np.isnan(correlation_with_target):
                        interaction_data.append(
                            {
                                "Feature1": feat1,
                                "Feature2": feat2,
                                "Interaction_Corr": correlation_with_target,
                                "Individual_Corr_Sum": correlations[feat1]
                                + correlations[feat2],
                            }
                        )
                        feature_pairs.append(f"{feat1} × {feat2}")

        interaction_df = pd.DataFrame(interaction_data)
        interaction_df = interaction_df.sort_values(
            "Interaction_Corr", key=abs, ascending=False
        )

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Top interaction correlations
        top_interactions = interaction_df.head(15)
        colors = [
            "red" if x < 0 else "green" for x in top_interactions["Interaction_Corr"]
        ]

        y_pos = range(len(top_interactions))
        ax1.barh(y_pos, top_interactions["Interaction_Corr"], color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(
            [
                f"{row['Feature1']} × {row['Feature2']}"
                for _, row in top_interactions.iterrows()
            ]
        )
        ax1.set_xlabel("Correlation with SalePrice")
        ax1.set_title("Top 15 Feature Interactions")
        ax1.grid(axis="x", alpha=0.3)

        # Interaction vs Individual correlations scatter plot
        ax2.scatter(
            interaction_df["Individual_Corr_Sum"],
            interaction_df["Interaction_Corr"],
            alpha=0.6,
            s=50,
        )
        ax2.set_xlabel("Sum of Individual Correlations")
        ax2.set_ylabel("Interaction Correlation")
        ax2.set_title("Feature Interactions vs Individual Effects")
        ax2.grid(alpha=0.3)

        # Add diagonal line for reference
        min_val = min(
            interaction_df["Individual_Corr_Sum"].min(),
            interaction_df["Interaction_Corr"].min(),
        )
        max_val = max(
            interaction_df["Individual_Corr_Sum"].max(),
            interaction_df["Interaction_Corr"].max(),
        )
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Print top interactions
        print("\n" + "=" * 60)
        print("FEATURE INTERACTION ANALYSIS")
        print("=" * 60)
        print(f"\nTop 10 Feature Interactions:")

        for _, row in interaction_df.head(10).iterrows():
            print(
                f"  {row['Feature1']} × {row['Feature2']}: {row['Interaction_Corr']:.4f}"
            )

        # Identify synergistic vs antagonistic interactions
        synergistic = interaction_df[
            interaction_df["Interaction_Corr"].abs()
            > interaction_df["Individual_Corr_Sum"].abs()
        ]
        if len(synergistic) > 0:
            print(f"\n🔄 Synergistic Interactions (stronger together):")
            for _, row in synergistic.head(5).iterrows():
                print(
                    f"  {row['Feature1']} × {row['Feature2']}: Interaction={row['Interaction_Corr']:.4f}, Individual Sum={row['Individual_Corr_Sum']:.4f}"
                )

        return interaction_df

    def perform_pca_analysis(self, n_components=10, figsize=(20, 12)):
        """Perform Principal Component Analysis to understand feature relationships"""
        # Prepare data for PCA
        pca_data = self.df[self.numerical_features].fillna(
            self.df[self.numerical_features].mean()
        )

        # Standardize the data
        scaled_data = self.scaler.fit_transform(pca_data)

        # Perform PCA
        pca = PCA(n_components=min(n_components, len(self.numerical_features)))
        pca_result = pca.fit_transform(scaled_data)

        # Create PCA DataFrame
        pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns)
        pca_df[self.target] = self.df[self.target].values

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Explained variance ratio
        ax1.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
        )
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("PCA - Explained Variance by Component")
        ax1.grid(axis="y", alpha=0.3)

        # Add cumulative variance line
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(1, len(cumsum) + 1), cumsum, "ro-", color="red", alpha=0.7)
        ax1_twin.set_ylabel("Cumulative Explained Variance", color="red")

        # 2. PC1 vs PC2 colored by target
        scatter = ax2.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df[self.target],
            cmap="viridis",
            alpha=0.6,
        )
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax2.set_title("PCA - First Two Components")
        plt.colorbar(scatter, ax=ax2, label=self.target)

        # 3. Feature loadings heatmap
        if pca.n_components_ >= 5:
            loadings = pca.components_[:5, :].T
            feature_names = self.numerical_features[: len(loadings)]

            sns.heatmap(
                loadings,
                xticklabels=[f"PC{i+1}" for i in range(5)],
                yticklabels=feature_names[: len(loadings)],
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=ax3,
                cbar_kws={"shrink": 0.8},
            )
            ax3.set_title("Feature Loadings (First 5 PCs)")

        # 4. PC1 correlation with target
        pc1_target_corr = pca_df["PC1"].corr(pca_df[self.target])
        ax4.scatter(pca_df["PC1"], pca_df[self.target], alpha=0.6)
        ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax4.set_ylabel(self.target)
        ax4.set_title(f"PC1 vs {self.target} (r={pc1_target_corr:.3f})")
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print PCA summary
        print("\n" + "=" * 60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 60)
        print(f"Number of components: {pca.n_components_}")
        print(
            f"Total variance explained by {pca.n_components_} components: {sum(pca.explained_variance_ratio_):.2%}"
        )

        print(f"\nVariance explained by each component:")
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var_ratio:.2%}")

        # Top feature loadings for each PC
        print(f"\nTop 5 feature loadings for first 3 PCs:")
        for i in range(min(3, pca.n_components_)):
            pc_loadings = pd.Series(pca.components_[i], index=self.numerical_features)
            top_loadings = pc_loadings.abs().sort_values(ascending=False).head(5)

            print(f"\n  PC{i+1}:")
            for feature, loading in top_loadings.items():
                original_loading = pc_loadings[feature]
                print(f"    {feature}: {original_loading:.3f}")

        return pca, pca_df

    def analyze_feature_clusters(self, n_clusters=5, figsize=(20, 10)):
        """Cluster features to identify groups of similar features"""
        # Create correlation matrix for clustering
        corr_matrix = self.df[self.numerical_features].corr()

        # Use 1 - |correlation| as distance measure
        distance_matrix = 1 - np.abs(corr_matrix)

        # Perform clustering on features
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.cluster import AgglomerativeClustering

        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method="ward")

        # K-means clustering on correlation matrix
        corr_values = corr_matrix.values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        feature_clusters = kmeans.fit_predict(corr_values)

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Dendrogram
        dendrogram(
            linkage_matrix,
            labels=self.numerical_features[: len(linkage_matrix) + 1],
            ax=ax1,
            orientation="top",
            leaf_rotation=90,
        )
        ax1.set_title("Feature Clustering Dendrogram")
        ax1.tick_params(axis="x", labelsize=8)

        # 2. Clustered correlation heatmap
        # Reorder features by clusters
        cluster_order = np.argsort(feature_clusters)
        ordered_features = [self.numerical_features[i] for i in cluster_order]
        ordered_corr = corr_matrix.loc[ordered_features, ordered_features]

        sns.heatmap(
            ordered_corr,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax2,
            cbar_kws={"shrink": 0.8},
        )
        ax2.set_title("Correlation Matrix (Clustered)")
        ax2.tick_params(axis="both", labelsize=8)

        # 3. Cluster composition
        cluster_info = pd.DataFrame(
            {"Feature": self.numerical_features, "Cluster": feature_clusters}
        )

        cluster_counts = cluster_info["Cluster"].value_counts().sort_index()
        ax3.bar(cluster_counts.index, cluster_counts.values)
        ax3.set_xlabel("Cluster")
        ax3.set_ylabel("Number of Features")
        ax3.set_title("Features per Cluster")
        ax3.grid(axis="y", alpha=0.3)

        # 4. Cluster centroids correlation with target
        target_correlations = []
        cluster_labels = []

        for cluster_id in range(n_clusters):
            cluster_features = cluster_info[cluster_info["Cluster"] == cluster_id][
                "Feature"
            ].tolist()
            if cluster_features:
                # Calculate mean correlation with target for cluster
                cluster_target_corr = (
                    self.df[cluster_features + [self.target]]
                    .corr()[self.target]
                    .drop(self.target)
                    .mean()
                )
                target_correlations.append(cluster_target_corr)
                cluster_labels.append(f"Cluster {cluster_id}")

        colors = ["red" if x < 0 else "green" for x in target_correlations]
        ax4.bar(cluster_labels, target_correlations, color=colors)
        ax4.set_ylabel(f"Mean Correlation with {self.target}")
        ax4.set_title(f"Cluster Average Correlation with {self.target}")
        ax4.grid(axis="y", alpha=0.3)
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        # Print cluster analysis
        print("\n" + "=" * 60)
        print("FEATURE CLUSTERING ANALYSIS")
        print("=" * 60)

        for cluster_id in range(n_clusters):
            cluster_features = cluster_info[cluster_info["Cluster"] == cluster_id][
                "Feature"
            ].tolist()
            print(f"\nCluster {cluster_id} ({len(cluster_features)} features):")

            if cluster_features:
                # Calculate internal cluster correlation
                if len(cluster_features) > 1:
                    cluster_corr_matrix = corr_matrix.loc[
                        cluster_features, cluster_features
                    ]
                    # Get upper triangle correlations
                    upper_corr = cluster_corr_matrix.where(
                        np.triu(np.ones(cluster_corr_matrix.shape), k=1).astype(bool)
                    )
                    mean_internal_corr = upper_corr.stack().mean()
                    print(f"  Mean internal correlation: {mean_internal_corr:.3f}")

                # Mean correlation with target
                cluster_target_corr = (
                    self.df[cluster_features + [self.target]]
                    .corr()[self.target]
                    .drop(self.target)
                )
                mean_target_corr = cluster_target_corr.mean()
                print(f"  Mean correlation with {self.target}: {mean_target_corr:.3f}")

                print(f"  Features: {', '.join(cluster_features)}")

        return cluster_info

    def analyze_dimensionality_reduction(
        self, methods=["PCA", "TSNE"], figsize=(20, 8)
    ):
        """Compare different dimensionality reduction techniques"""
        # Prepare data
        analysis_data = self.df[self.numerical_features].fillna(
            self.df[self.numerical_features].mean()
        )
        scaled_data = self.scaler.fit_transform(analysis_data)

        results = {}

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)
        if len(methods) == 1:
            axes = [axes]

        for i, method in enumerate(methods):
            if method == "PCA":
                # PCA
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(scaled_data)
                explained_var = sum(pca.explained_variance_ratio_)
                results[method] = {
                    "data": reduced_data,
                    "explained_variance": explained_var,
                    "components": pca.components_,
                }
                title = f"PCA (Explained Var: {explained_var:.2%})"

            elif method == "TSNE":
                # t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                reduced_data = tsne.fit_transform(scaled_data)
                results[method] = {"data": reduced_data}
                title = "t-SNE"

            # Plot
            scatter = axes[i].scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                c=self.df[self.target],
                cmap="viridis",
                alpha=0.6,
            )
            axes[i].set_title(title)
            axes[i].set_xlabel(f"{method} Component 1")
            axes[i].set_ylabel(f"{method} Component 2")
            plt.colorbar(scatter, ax=axes[i], label=self.target)

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 60)
        print("DIMENSIONALITY REDUCTION COMPARISON")
        print("=" * 60)

        for method in methods:
            print(f"\n{method}:")
            if method == "PCA":
                print(
                    f"  Explained variance: {results[method]['explained_variance']:.2%}"
                )
                print(f"  Information preserved in 2D projection")
            elif method == "TSNE":
                print(f"  Non-linear projection preserving local structure")

        return results

    def analyze_feature_importance_combinations(self, top_k=15, figsize=(15, 10)):
        """Analyze feature importance using multiple selection methods"""
        # Prepare data
        X = self.df[self.numerical_features].fillna(
            self.df[self.numerical_features].mean()
        )
        y = self.df[self.target]

        # Method 1: Correlation-based
        correlations = X.corrwith(y).abs().sort_values(ascending=False)

        # Method 2: Univariate statistical test
        selector = SelectKBest(score_func=f_regression, k="all")
        selector.fit(X, y)
        f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(
            ascending=False
        )

        # Method 3: Mutual information (approximated by correlation for numerical)
        from sklearn.feature_selection import mutual_info_regression

        mi_scores = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # Combine methods
        feature_scores = pd.DataFrame(
            {"Correlation": correlations, "F_Score": f_scores, "Mutual_Info": mi_scores}
        )

        # Normalize scores
        for col in feature_scores.columns:
            feature_scores[f"{col}_norm"] = (
                feature_scores[col] - feature_scores[col].min()
            ) / (feature_scores[col].max() - feature_scores[col].min())

        # Calculate combined score
        feature_scores["Combined_Score"] = (
            feature_scores["Correlation_norm"]
            + feature_scores["F_Score_norm"]
            + feature_scores["Mutual_Info_norm"]
        ) / 3

        # Sort by combined score
        feature_scores = feature_scores.sort_values("Combined_Score", ascending=False)

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Top features by different methods
        top_features = feature_scores.head(top_k)

        x_pos = np.arange(len(top_features))
        width = 0.25

        ax1.bar(
            x_pos - width,
            top_features["Correlation_norm"],
            width,
            label="Correlation",
            alpha=0.8,
        )
        ax1.bar(x_pos, top_features["F_Score_norm"], width, label="F-Score", alpha=0.8)
        ax1.bar(
            x_pos + width,
            top_features["Mutual_Info_norm"],
            width,
            label="Mutual Info",
            alpha=0.8,
        )

        ax1.set_xlabel("Features")
        ax1.set_ylabel("Normalized Score")
        ax1.set_title(f"Top {top_k} Features by Different Methods")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(top_features.index, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Combined score ranking
        ax2.barh(
            range(len(top_features)),
            top_features["Combined_Score"],
            color="darkblue",
            alpha=0.7,
        )
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features.index)
        ax2.set_xlabel("Combined Score")
        ax2.set_title(f"Top {top_k} Features - Combined Score")
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print feature importance analysis
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        print(f"\nTop {min(10, top_k)} Features by Combined Score:")

        for i, (feature, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"  {i:2d}. {feature}:")
            print(f"      Combined Score: {row['Combined_Score']:.3f}")
            print(f"      Correlation: {row['Correlation']:.3f}")
            print(f"      F-Score: {row['F_Score']:.1f}")
            print(f"      Mutual Info: {row['Mutual_Info']:.3f}")

        # Agreement analysis
        print(f"\nMethod Agreement Analysis:")
        top_10_correlation = set(correlations.head(10).index)
        top_10_f_score = set(f_scores.head(10).index)
        top_10_mi = set(mi_scores.head(10).index)

        all_methods_agree = top_10_correlation & top_10_f_score & top_10_mi
        two_methods_agree = (
            (top_10_correlation & top_10_f_score)
            | (top_10_correlation & top_10_mi)
            | (top_10_f_score & top_10_mi)
        )

        print(f"  Features in top 10 by all methods: {len(all_methods_agree)}")
        if all_methods_agree:
            print(f"    {', '.join(sorted(all_methods_agree))}")

        print(f"  Features in top 10 by at least 2 methods: {len(two_methods_agree)}")

        return feature_scores

    def __call__(self, analysis_type="all"):
        """Main method to run multivariate analysis"""
        print("\n" + "=" * 60)
        print("MULTIVARIATE ANALYSIS - HOUSE PRICE PREDICTION")
        print("=" * 60)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")

        results = {}

        if analysis_type in ["all", "interactions"]:
            print("\n🔄 Analyzing Feature Interactions...")
            results["interactions"] = self.analyze_feature_interactions()

        if analysis_type in ["all", "pca"]:
            print("\n📊 Performing PCA Analysis...")
            pca, pca_df = self.perform_pca_analysis()
            results["pca"] = {"model": pca, "data": pca_df}

        if analysis_type in ["all", "clustering"]:
            print("\n🔗 Analyzing Feature Clusters...")
            results["clusters"] = self.analyze_feature_clusters()

        if analysis_type in ["all", "dimensionality"]:
            print("\n📈 Comparing Dimensionality Reduction...")
            results["dimensionality"] = self.analyze_dimensionality_reduction()

        if analysis_type in ["all", "importance"]:
            print("\n⭐ Analyzing Feature Importance...")
            results["importance"] = self.analyze_feature_importance_combinations()

        print("\n✅ Multivariate Analysis Complete!")
        return results


# Create instance for easy usage
multivariate_analyzer = MultivariateAnalyzer(train_df)

if __name__ == "__main__":
    # Run complete analysis
    results = multivariate_analyzer()
    print("\n📋 Analysis Results Summary:")
    print(f"Available results: {list(results.keys())}")
