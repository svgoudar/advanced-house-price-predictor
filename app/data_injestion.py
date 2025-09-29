import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import os
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from kagglehub.config import set_kaggle_credentials, clear_kaggle_credentials
import kagglehub
# set_kaggle_credentials(username="sanjeevvgoudar", api_key="794736c3ca28d279a0297c91d6aee476")

# clear_kaggle_credentials()

# os.environ["KAGGLE_USERNAME"] = "sanjeevvgoudar"
# os.environ["KAGGLE_KEY"] = "794736c3ca28d279a0297c91d6aee476"

# print(os.environ.items())
# kagglehub.login()
house_prices_advanced_regression_techniques_path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
print('Data source import complete.')


train_df = pd.read_csv(os.path.join(house_prices_advanced_regression_techniques_path, "train.csv"))

test_df = pd.read_csv(os.path.join(house_prices_advanced_regression_techniques_path, "test.csv"))

sample_submission = pd.read_csv(os.path.join(house_prices_advanced_regression_techniques_path, "sample_submission.csv"))