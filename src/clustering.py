import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_segmentation(df, X_scaled, n_clusters=5):
    """
    Evaluates cluster counts K=2..8 and fits K-Means model on actual data.
    Generates data-driven persona names based on empirical centroids.
    Uses new Cyan/Aqua/Cream design palette.
    """
    elbow_data = []
    best_k = n_clusters
    best_sil = -1.0
    
    for k in range(2, 9):
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_test = km_test.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels_test))
        inertia = float(km_test.inertia_)
        
        elbow_data.append({
            'k': k,
            'inertia': round(inertia, 2),
            'silhouette': round(sil, 4)
        })
        if sil > best_sil:
            best_sil = sil
            
    # Fit final model with chosen K=5
    final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X_scaled)
    df_result = df.copy()
    df_result['Cluster'] = cluster_labels
    
    # Calculate empirical centroids
    cluster_stats = df_result.groupby('Cluster').agg(
        avg_income=('Annual Income (k$)', 'mean'),
        avg_spending=('Spending Score (1-100)', 'mean'),
        avg_age=('Age', 'mean'),
        count=('CustomerID', 'count')
    ).reset_index()
    
    profiles = []
    total_cust = len(df_result)
    
    for _, row in cluster_stats.iterrows():
        c_id = int(row['Cluster'])
        inc = float(round(row['avg_income'], 1))
        sp = float(round(row['avg_spending'], 1))
        age = float(round(row['avg_age'], 1))
        cnt = int(row['count'])
        pct = float(round(cnt / total_cust * 100, 1))
        
        # Interpret dynamic persona title from actual income and spending score
        # Using new Palette (#69D2E7 Primary Cyan, #46b5cc Deep Aqua, #059669 Emerald, #d97706 Amber, #4b5563 Slate)
        if inc >= 70 and sp >= 60:
            name = "High-Income High-Spenders"
            tag = "High Income • High Spend"
            strategy = "VIP concierge services, premium product previews, exclusive loyalty rewards"
            color = "#69D2E7" # Primary Cyan
        elif inc >= 70 and sp < 60:
            name = "High-Income Low-Spenders"
            tag = "High Income • Low Spend"
            strategy = "Targeted quality reassurance, premium brand positioning, luxury cross-sells"
            color = "#46b5cc" # Deep Aqua
        elif inc < 45 and sp >= 60:
            name = "Low-Income High-Spenders"
            tag = "Low Income • High Spend"
            strategy = "BNPL payment options, limited-time flash deals, trending social promotions"
            color = "#059669" # Emerald Teal
        elif inc < 45 and sp < 60:
            name = "Low-Income Low-Spenders"
            tag = "Low Income • Low Spend"
            strategy = "Budget bundle discounts, clearance notifications, high-value entry packs"
            color = "#d97706" # Warm Amber
        else:
            name = "Moderate-Income Balanced Shoppers"
            tag = "Moderate Income • Moderate Spend"
            strategy = "Seasonal promotions, personalized recommendations, rewards point incentives"
            color = "#4b5563" # Slate Charcoal
            
        profiles.append({
            'cluster_id': c_id,
            'name': f"Cluster {c_id}: {name}",
            'short_name': name,
            'tag': tag,
            'color': color,
            'strategy': strategy,
            'count': cnt,
            'pct': pct,
            'avg_income': inc,
            'avg_spending_score': sp,
            'avg_age': age
        })
        
    return final_kmeans, df_result, elbow_data, profiles
