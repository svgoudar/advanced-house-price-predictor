import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------- Mixins -----------------

class MissingStatsMixin:
    def compute_missing_summary(self, df):
        self.missing_summary = df.isnull().sum().sort_values(ascending=False)
        print("Missing Values Summary:")
        print(self.missing_summary[self.missing_summary > 0].to_list())


class ColumnCategorizerMixin:
    def categorize_columns(self, df, low_card_threshold=10):
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        self.low_card_cols = [
            c for c in cat_cols if df[c].nunique() <= low_card_threshold
        ]
        self.high_card_cols = [
            c for c in cat_cols if df[c].nunique() > low_card_threshold
        ]


class MissingVisualizerMixin:
    def visualize_missing(self, df, figsize=(12, 6), cmap="viridis"):
        # Heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(df.isnull(), cbar=False, cmap=cmap)
        plt.title("Missing Values Heatmap")
        plt.show()

        # Bar plot
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        if not missing_counts.empty:
            plt.figure(figsize=(12, 5))
            sns.barplot(
                x=missing_counts.index, 
                y=missing_counts.values, 
                palette="viridis"
            )
            plt.ylabel("Number of Missing Values")
            plt.title("Missing Values Count per Feature")
            plt.xticks(rotation=45)
            plt.show()
