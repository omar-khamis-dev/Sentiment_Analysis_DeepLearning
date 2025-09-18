import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, text_col="text", label_col="label", test_size=0.2, random_state=42):
    """
    Load data from a CSV file and split into train and test sets.

    Args:
        path (str): Path to the CSV file.
        text_col (str): Column name containing the text data.
        label_col (str): Column name containing the labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test (DataFrames / Series)
    """
    df = pd.read_csv(path)
    
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"The dataset must contain columns: {text_col}, {label_col}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]
    )

    return X_train, X_test, y_train, y_test