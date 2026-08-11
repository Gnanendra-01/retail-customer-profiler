import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_clustering_data(df):
    """
    Extracts primary 2D features for Mall Customer K-Means segmentation:
    - Annual Income (k$)
    - Spending Score (1-100)
    Returns scaled array and fit StandardScaler instance.
    """
    feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    X_raw = df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return X_scaled, scaler, feature_cols

def preprocess_classification_data(df, spending_threshold=70):
    """
    Defines High-Value Target and features WITHOUT target leakage:
    - Target: IsHighValue = 1 if Spending Score (1-100) >= spending_threshold
    - Features: ['Age', 'Annual Income (k$)', 'Gender_Male']
      (Spending Score is explicitly EXCLUDED to prevent target leakage!)
    """
    df_clf = df.copy()
    
    # Construct Target Variable
    df_clf['IsHighValue'] = (df_clf['Spending Score (1-100)'] >= spending_threshold).astype(int)
    
    # Categorical encoding for Gender
    df_clf['Gender_Male'] = (df_clf['Gender'] == 'Male').astype(int)
    
    feature_cols = ['Age', 'Annual Income (k$)', 'Gender_Male']
    X = df_clf[feature_cols]
    y = df_clf['IsHighValue']
    
    return X, y, df_clf, feature_cols
