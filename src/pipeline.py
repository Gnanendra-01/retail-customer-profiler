import os
import json
import pickle
import numpy as np
import pandas as pd

from data_loader import load_mall_customers
from preprocessing import preprocess_clustering_data, preprocess_classification_data
from clustering import run_kmeans_segmentation
from classification import train_high_value_classifier

def run_pipeline():
    print("=" * 60)
    print("RUNNING RETAIL ANALYTICS ML PIPELINE (MALL_CUSTOMERS.CSV)")
    print("=" * 60)
    
    # 1. Load actual raw dataset
    df, csv_path = load_mall_customers()
    print(f"[1/5] Loaded {len(df)} records from '{csv_path}'")
    
    # 2. Preprocess for Clustering & Classification
    X_cluster_scaled, scaler_cluster, cluster_features = preprocess_clustering_data(df)
    X_clf, y_clf, df_clf, clf_features = preprocess_classification_data(df, spending_threshold=70)
    print(f"[2/5] Preprocessed clustering features ({cluster_features}) and classification features ({clf_features})")
    
    # 3. K-Means Segmentation
    kmeans_model, df_segmented, elbow_data, cluster_profiles = run_kmeans_segmentation(df_clf, X_cluster_scaled, n_clusters=5)
    print(f"[3/5] K-Means clustering completed. Silhouette score for K=5: {elbow_data[3]['silhouette']}")
    
    # 4. Supervised Classifier
    clf_model, clf_metrics, feature_importances = train_high_value_classifier(X_clf, y_clf)
    print(f"[4/5] Random Forest Classifier trained. Accuracy: {clf_metrics['accuracy']}%, ROC-AUC: {clf_metrics['auc']}")
    
    # 5. Save pickle model artifacts into models/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'clustering_model.pkl'), 'wb') as f:
        pickle.dump(kmeans_model, f)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_cluster, f)
    with open(os.path.join(models_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(clf_model, f)
    print(f"[5/5] Model artifacts saved to '{models_dir}'")
    
    # 6. Extract actual customer records for web visualization
    customer_records = []
    for _, row in df_segmented.iterrows():
        customer_records.append({
            'id': int(row['CustomerID']),
            'gender': str(row['Gender']),
            'age': int(row['Age']),
            'income': float(row['Annual Income (k$)']),
            'spending_score': int(row['Spending Score (1-100)']),
            'cluster_id': int(row['Cluster']),
            'is_high_value': int(row['IsHighValue'])
        })
        
    scaler_means = scaler_cluster.mean_.tolist()
    scaler_stds = scaler_cluster.scale_.tolist()
    
    output_data = {
        'total_customers': int(len(df)),
        'avg_age': float(round(df['Age'].mean(), 2)),
        'avg_income': float(round(df['Annual Income (k$)'].mean(), 2)),
        'avg_spending': float(round(df['Spending Score (1-100)'].mean(), 2)),
        'min_income': float(df['Annual Income (k$)'].min()),
        'max_income': float(df['Annual Income (k$)'].max()),
        'min_spending': int(df['Spending Score (1-100)'].min()),
        'max_spending': int(df['Spending Score (1-100)'].max()),
        'high_value_count': int(y_clf.sum()),
        'high_value_pct': float(round(y_clf.mean() * 100, 1)),
        'elbow_data': elbow_data,
        'cluster_profiles': cluster_profiles,
        'feature_importances': feature_importances,
        'metrics': clf_metrics,
        'scaler': {
            'means': scaler_means,
            'stds': scaler_stds
        },
        'customer_records': customer_records
    }
    
    web_json_path = os.path.join(project_root, 'src', 'web', 'model_data.json')
    os.makedirs(os.path.dirname(web_json_path), exist_ok=True)
    with open(web_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Exported model_data.json successfully to '{web_json_path}'")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
