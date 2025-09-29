import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from utils.core import BaseAnalyzer


class MissingValueAnalyzer(BaseAnalyzer):
    """
    Analyze and visualize missing values in a DataFrame.
    """

    def __init__(self, low_card_threshold=10):
        self.low_card_threshold = low_card_threshold
        self.num_cols = []
        self.low_card_cols = []
        self.high_card_cols = []
        self.sparse_columns = []

    def analyze(self, df):
        """
        Analyze missing values and categorize features.
        """
        # Missing values summary
        self.missing_summary = df.isnull().sum().sort_values(ascending=False)


        # Identify column types
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        self.low_card_cols = [
            c for c in cat_cols if df[c].nunique() <= self.low_card_threshold
        ]
        self.high_card_cols = [
            c for c in cat_cols if df[c].nunique() > self.low_card_threshold
        ]

    def summary(self):
        cols_with_missing = self.data.columns[self.data.isnull().sum() > 0]

        report = pd.DataFrame(
            {
                "ColumnName": cols_with_missing,
                "MissingValues": self.data[cols_with_missing].isnull().sum().values,
                "MissingPercentage": round(
                    self.data[cols_with_missing].isnull().mean() * 100, 2
                ),
                "Dtype": self.data[cols_with_missing].dtypes.values,
                "Values": [
                    self.data[col].value_counts(dropna=True).to_dict()
                    for col in cols_with_missing
                ],
            }
        )

        report = report.sort_values(
            by="MissingPercentage", ascending=False
        )

        def decide_strategy(row):
            pct = row["MissingPercentage"]

            if pct > 80:
                return "Drop Column"
            return "Impute"

        report["Strategy"] = report.apply(decide_strategy, axis=1)

        self.sparse_columns = (
            report["ColumnName"]
            .where(report["Strategy"] == "Drop Column")
            .dropna()
            .tolist()
        )
        return report

    def visualize(self, figsize=(12, 6), cmap="viridis"):
        """
        Visualize missing values as heatmap and bar chart.
        """
        # Heatmap
        self.log("Visualizing Missing Percentage")

        missing_report = self.summary()
        if not missing_report.empty:
            plt.figure(figsize=figsize)
            sns.barplot(x="MissingPercentage", y=missing_report.index, data=missing_report,palette=cmap)
            plt.ylabel("Number of Missing Values")
            plt.title("Missing Values Count per Feature")
            plt.xticks(rotation=45)
            plt.show()

    # Make the class callable
    def __call__(self, df):
        setattr(self, "data", df)
        self.analyze(df)
        self.visualize()
        return self.sparse_columns, self.data

    # Iterator over features with missing values
    def __iter__(self):
        for col in self.missing_summary[self.missing_summary > 0].index:
            yield col

    def __repr__(self):
        return f"<MissingValueAnalyzer(low_card_threshold={self.low_card_threshold})>"
