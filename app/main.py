from data_injestion import train_df
from preprocessor import MissingValueAnalyzer


if __name__ == "__main__":

    analyzer = MissingValueAnalyzer(low_card_threshold=2)

# Analyze + visualize in one call
    analyzer(train_df)

# Iterate over columns with missing values
    for col in analyzer:
        print("Feature with missing values:", col)

