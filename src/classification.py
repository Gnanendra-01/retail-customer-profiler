import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

def train_high_value_classifier(X, y):
    """
    Trains Random Forest Classifier to predict High-Value Customer status.
    Calculates genuine test set metrics and feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    importances = [
        {
            "feature": col,
            "importance": float(round(imp, 4))
        }
        for col, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    ]
    
    metrics = {
        'accuracy': float(round(acc * 100, 1)),
        'auc': float(round(auc, 4)),
        'confusion_matrix': cm,
        'test_size': len(y_test),
        'train_size': len(y_train)
    }
    
    return rf, metrics, importances
