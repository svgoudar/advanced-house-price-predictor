from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class SkewedImputerMixin:
    def _numeric_imputer_strategy(self, df, skew_threshold=0.5):
        """Determine imputation strategy based on skewness for numerical columns"""
        # Exclude Id column if present
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "Id" in numeric_cols:
            numeric_cols.remove("Id")

        self.numeric_strategies = {}
        for col in numeric_cols:
            # Handle columns with all NaN values
            if df[col].notna().sum() == 0:
                self.numeric_strategies[col] = "median"
                continue

            skewness = df[col].skew()
            # Use median for skewed data, mean for normal distribution
            strategy = "median" if abs(skewness) > skew_threshold else "mean"
            self.numeric_strategies[col] = strategy
        return numeric_cols


class MissingValueAnalyzerMixin:
    def summary(self, df):
        """Analyze missing values and determine handling strategy"""
        self.data = df.copy()
        cols_with_missing = df.columns[df.isnull().sum() > 0]

        if len(cols_with_missing) == 0:
            print("No missing values found in the dataset.")
            self.sparse_columns = []
            return pd.DataFrame()

        report = pd.DataFrame(
            {
                "ColumnName": cols_with_missing,
                "MissingValues": df[cols_with_missing].isnull().sum().values,
                "MissingPercentage": round(
                    df[cols_with_missing].isnull().mean() * 100, 2
                ),
                "Dtype": df[cols_with_missing].dtypes.values,
                "UniqueValues": [
                    df[col].nunique(dropna=True) for col in cols_with_missing
                ],
            }
        ).sort_values(by="MissingPercentage", ascending=False)

        # Decide strategy based on missing percentage
        report["Strategy"] = report["MissingPercentage"].apply(
            lambda pct: "Drop Column" if pct > 80 else "Impute"
        )

        # Store columns to be dropped
        self.sparse_columns = report.loc[
            report["Strategy"] == "Drop Column", "ColumnName"
        ].tolist()

        print(f"Found {len(cols_with_missing)} columns with missing values.")
        print(f"Columns to drop (>80% missing): {len(self.sparse_columns)}")

        return report


class ImputerPipelineMixin:
    def build_imputer_pipeline(self, df, cardinality_threshold=10, skew_threshold=0.5):
        """Build preprocessing pipeline with appropriate imputers for different column types"""
        # Drop sparse columns (>80% missing)
        df_clean = df.drop(columns=getattr(self, "sparse_columns", []), errors="ignore")
        print(f"Dropped {len(getattr(self, 'sparse_columns', []))} sparse columns")

        # Get numeric columns and their strategies
        numeric_cols = self._numeric_imputer_strategy(df_clean, skew_threshold)
        numeric_strategies = getattr(self, "numeric_strategies", {})

        # Get categorical columns and split by cardinality
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        low_card_cols = [
            col
            for col in cat_cols
            if df_clean[col].nunique(dropna=False) <= cardinality_threshold
        ]
        high_card_cols = [
            col
            for col in cat_cols
            if df_clean[col].nunique(dropna=False) > cardinality_threshold
        ]

        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Low cardinality categorical: {len(low_card_cols)}")
        print(f"High cardinality categorical: {len(high_card_cols)}")

        # Build transformers list
        transformers = []

        # Numeric transformer with skew-aware imputation
        if numeric_cols:

            def skewed_imputer(X):
                """Custom imputer that handles skewed numeric data"""
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)

                X_df = pd.DataFrame(X, columns=numeric_cols[: X.shape[1]])

                for i, col in enumerate(numeric_cols[: X.shape[1]]):
                    strategy = numeric_strategies.get(col, "median")
                    imputer = SimpleImputer(strategy=strategy)
                    X_df.iloc[:, [i]] = imputer.fit_transform(X_df.iloc[:, [i]])

                return X_df.values

            numeric_pipeline = FunctionTransformer(skewed_imputer, validate=False)
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        # Low cardinality categorical transformer
        if low_card_cols:
            low_card_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="most_frequent"))]
            )
            transformers.append(("low_card", low_card_pipeline, low_card_cols))

        # High cardinality categorical transformer
        if high_card_cols:
            high_card_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))]
            )
            transformers.append(("high_card", high_card_pipeline, high_card_cols))

        # Create ColumnTransformer
        if not transformers:
            raise ValueError("No valid columns found for preprocessing")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any remaining columns
            sparse_threshold=0,  # Return dense array
        )

        # Store pipeline information
        self.pipeline_info = {
            "numeric": numeric_cols,
            "low_card": low_card_cols,
            "high_card": high_card_cols,
            "dropped": getattr(self, "sparse_columns", []),
            "all_processed": numeric_cols + low_card_cols + high_card_cols,
        }

        return preprocessor


class DataPreprocessor(
    SkewedImputerMixin,
    MissingValueAnalyzerMixin,
    ImputerPipelineMixin,
    BaseEstimator,
    TransformerMixin,
):
    def __init__(self, cardinality_threshold=10, skew_threshold=0.5, sparse_columns=[]):
        self.cardinality_threshold = cardinality_threshold
        self.skew_threshold = skew_threshold
        self.sparse_columns = sparse_columns

    def fit(self, X, y=None):
        """Fit the preprocessor to the training data"""
        if self.sparse_columns:
            X.drop(columns=self.sparse_columns, inplace=True, axis=1)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        print(f"🔄 Fitting DataPreprocessor on data with shape: {X.shape}")

        try:
            # Step 1: Analyze missing values and identify sparse columns
            print("Step 1: Analyzing missing values...")
            missing_report = self.summary(X)

            # Step 2: Determine numeric imputation strategies based on skewness
            print("Step 2: Determining numeric imputation strategies...")
            self._numeric_imputer_strategy(X, self.skew_threshold)

            # Step 3: Build and fit the preprocessing pipeline
            print("Step 3: Building preprocessing pipeline...")
            self.imputer_pipeline = self.build_imputer_pipeline(
                X, self.cardinality_threshold, self.skew_threshold
            )

            print("Step 4: Fitting pipeline...")
            self.imputer_pipeline.fit(X)

            print("✅ DataPreprocessor fitted successfully!")
            return self

        except Exception as e:
            print(f"❌ Error during fitting: {e}")
            raise

    def transform(self, X):
        """Transform the input data using the fitted pipeline"""
        if not hasattr(self, "imputer_pipeline"):
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Apply transformation
        X_transformed = self.imputer_pipeline.transform(X)

        # Get column names in the correct order
        all_cols = self.pipeline_info["all_processed"]

        # Validate dimensions
        if X_transformed.shape[1] != len(all_cols):
            print(f"Warning: Shape mismatch!")
            print(f"X_transformed shape: {X_transformed.shape}")
            print(f"Expected columns: {len(all_cols)}")
            print(f"Using generic column names...")
            # all_cols = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Convert to DataFrame
        try:
            X_transformed_df = pd.DataFrame(
                X_transformed, columns=all_cols, index=X.index
            )

            print(f"✅ Successfully transformed to DataFrame: {X_transformed_df.shape}")
            return X_transformed_df

        except Exception as e:
            print(f"❌ Error creating DataFrame: {e}")
            print(f"Returning numpy array instead")
            return X_transformed

    def get_feature_names_out(self):
        """Get the names of the output features after transformation"""
        if hasattr(self, "pipeline_info"):
            return self.pipeline_info["all_processed"]
        else:
            raise ValueError("Pipeline not fitted yet. Call fit() first.")

    def fit_transform(self, X, y=None):
        """Fit the transformer and transform the data in one step"""
        return self.fit(X, y).transform(X)

    def get_pipeline_summary(self):
        """Get a summary of the preprocessing pipeline"""
        if not hasattr(self, "pipeline_info"):
            return "Pipeline not fitted yet."

        summary = {
            "Total features processed": len(self.pipeline_info["all_processed"]),
            "Numeric features": len(self.pipeline_info["numeric"]),
            "Low cardinality categorical": len(self.pipeline_info["low_card"]),
            "High cardinality categorical": len(self.pipeline_info["high_card"]),
            "Dropped features": len(self.pipeline_info["dropped"]),
        }

        return summary
