# missing_analyzer = MissingValueAnalyzer()
# sparse_columns,df = missing_analyzer(train_df)

# preprocessor = Preprocessor(skew_threshold=0.5, cardinality_threshold=10,sparse_columns=sparse_columns)
# df_clean = preprocessor.fit_transform(df)

# # print(cleaned_df.dtypes)
# print(df_clean[df.columns].head(5))
# print(df_clean.isnull().sum())

# print(train_df[df.columns].head(5))
from preprocessor.missing_value_analysis import MissingValueAnalyzer
from preprocessor.imputer import Preprocessor
from data_injestion import train_df,test_df
from preprocessor.main import DataPreprocessor

sparse_columns = test_df.isnull().mean()[test_df.isnull().mean() > 0.8].index.to_list()
preprocessor = DataPreprocessor(cardinality_threshold=10, skew_threshold=0.5,sparse_columns=sparse_columns)
preprocessor.fit_transform(test_df)

print(preprocessor.summary(test_df))

df_imputed = preprocessor.transform(test_df)
df_imputed.head()