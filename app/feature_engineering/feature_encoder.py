import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# -----------------------------
# 1) Feature Groups
# -----------------------------
# ordinal_features = ["ExterQual", "BsmtQual", "KitchenQual"]
# categorical_features = ["Neighborhood", "GarageFinish", "Foundation"]
# numerical_features = ["OverallQual", "GrLivArea"]

# ordinal_mappings = {
#     "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
#     "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
#     "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
# }


# -----------------------------
# 5) Decode Helper (works with both Pipeline and make_pipeline)
# -----------------------------
def decode(pipeline, encoder_step, feature, encoded_value):
    """
    Recover original category from encoded integer.
    Works with both Pipeline and make_pipeline.

    For make_pipeline, step names are auto-generated like:
    - 'ordinalmapper' (lowercase, no underscores)
    - 'categoricallabelencoder'
    """
    # Handle both named steps (Pipeline) and auto-generated names (make_pipeline)
    if hasattr(pipeline, "named_steps"):
        # For Pipeline with named steps
        if encoder_step in pipeline.named_steps:
            encoders = pipeline.named_steps[encoder_step].encoders
        else:
            # For make_pipeline, try auto-generated names
            step_names = list(pipeline.named_steps.keys())
            print(f"Available steps: {step_names}")
            raise ValueError(
                f"Step '{encoder_step}' not found. Available steps: {step_names}"
            )
    else:
        # For very old sklearn versions or custom pipelines
        encoders = pipeline.steps[encoder_step][1].encoders

    if feature not in encoders:
        raise ValueError(f"Feature '{feature}' not found in encoders.")

    le = encoders[feature]
    if encoded_value < 0 or encoded_value >= len(le.classes_):
        raise ValueError(f"Encoded value {encoded_value} out-of-range for '{feature}'.")

    return le.inverse_transform([encoded_value])[0]


def decode_make_pipeline(pipeline, feature, encoded_value):
    """
    Simplified decode function specifically for make_pipeline.
    Automatically finds the CategoricalLabelEncoder step.
    """
    # Find the CategoricalLabelEncoder step
    encoder_step = None
    for step_name, step_obj in pipeline.named_steps.items():
        if isinstance(step_obj, CategoricalLabelEncoder):
            encoder_step = step_obj
            break

    if encoder_step is None:
        raise ValueError("CategoricalLabelEncoder not found in pipeline")

    if feature not in encoder_step.encoders:
        raise ValueError(f"Feature '{feature}' not found in encoders.")

    le = encoder_step.encoders[feature]
    if encoded_value < 0 or encoded_value >= len(le.classes_):
        raise ValueError(f"Encoded value {encoded_value} out-of-range for '{feature}'.")

    return le.inverse_transform([encoded_value])[0]


class OrdinalEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols, categories):
        self.cols = cols
        self.categories = categories
        self.encoder = OrdinalEncoder(categories=self.categories)

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        X_new = X.copy()
        print(self.encoder.transform(X[self.cols]))
        X_new[self.cols] = self.encoder.transform(X[self.cols])
        X_new[self.cols] = X_new[self.cols].astype(int)
        return X_new


class FrequencyEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        self.cols_ = self.cols or X.select_dtypes(include="object").columns.tolist()
        for col in self.cols_:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.cols_:
            X_new[col + "_freq"] = X_new[col].map(self.freq_maps_[col]).fillna(0)
        return X_new


# -------------------------
# 2. OneHot Encoder Wrapper
# -------------------------
class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_ = []

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        self.feature_names_ = self.encoder.get_feature_names_out(self.cols)
        return self

    def transform(self, X):
        X_new = X.copy()

        # Transform and get dense array
        data = self.encoder.transform(X_new[self.cols])

        # Debug information
        print(f"Transformed data shape: {data.shape}")
        print(f"Feature names: {self.feature_names_}")
        print(f"Number of feature names: {len(self.feature_names_)}")

        # Create DataFrame with encoded features
        encoded = pd.DataFrame(
            data,
            columns=self.feature_names_,
            index=X_new.index,
        )

        # Remove original columns and add encoded ones
        X_new = X_new.drop(columns=self.cols)
        X_new = pd.concat([X_new, encoded.apply(round).astype("int")], axis=1)
        return X_new


# -------------------------
# 3. Target Encoder Wrapper
# -------------------------
class DataFrameTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder for regression. Compatible with sklearn pipelines.
    Accepts X (DataFrame) and y (Series) in fit, returns DataFrame in transform.
    """

    def __init__(self, cols=None, target=None, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.target = target
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X):
        self.cols_ = self.cols or X.select_dtypes(include="object").columns.tolist()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if self.target is None:
            raise ValueError("y must be provided.")
        # if self.use_mapping:
        print("COLLL ", X.columns)
        if self.target in X.columns:
            y = X[self.target]
        else:
            return self

        self.global_mean_ = y.mean()

        for col in self.cols_:
            stats = X.groupby(col).size().to_frame("count")
            means = X.groupby(col).apply(lambda df: y[df.index].mean())
            stats["mean"] = means
            smoothing = 1 / (1 + np.exp(-(stats["count"] - 1) / self.smoothing))
            self.mapping_[col] = (
                self.global_mean_ * (1 - smoothing) + stats["mean"] * smoothing
            ).apply(round)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        X_new = X.copy()
        for col in self.cols_:
            X_new[col] = X_new[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_new


def create_encoder_pipeline(
    ordinal_categories,
    ordinal_features,
    low_cardinal_nominal_features,
    high_cardinal_nominal_features,
    target,
):

    pipe = Pipeline(
        [
            ("ordinal", OrdinalEncoderDF(ordinal_features, ordinal_categories)),
            ("low_card_ohe", OneHotEncoderDF(low_cardinal_nominal_features)),
            (
                "high_card_target",
                DataFrameTargetEncoder(high_cardinal_nominal_features, target=target),
            ),
        ]
    )

    return pipe
