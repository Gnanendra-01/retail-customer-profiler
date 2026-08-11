import os
import pandas as pd

def get_raw_csv_path():
    """
    Robust path resolution for Mall_Customers.csv relative to project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    csv_path = os.path.join(project_root, 'data', 'raw', 'Mall_Customers.csv')
    return csv_path

def load_mall_customers():
    """
    Loads raw Mall_Customers.csv directly.
    Fails with clear error if file does not exist or columns are invalid.
    DOES NOT fall back to synthetic data.
    """
    csv_path = get_raw_csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"REQUIRED DATASET MISSING: Could not find raw dataset at '{csv_path}'. "
            f"Please ensure Mall_Customers.csv is located in 'data/raw/' directory."
        )
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"INVALID DATASET SCHEMA: Missing required columns {missing_cols} in '{csv_path}'. "
            f"Expected schema columns: {required_cols}"
        )
        
    return df, csv_path

if __name__ == '__main__':
    df, path = load_mall_customers()
    print(f"Loaded {len(df)} records from {path}")
    print("Columns:", list(df.columns))
