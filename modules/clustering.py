import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def prepare_features_for_clustering(df, df_scaled, all_selected_features, selected_numerical, selected_categorical):
    """
    Prepare features for clustering by combining scaled numerical and one-hot encoded categorical features.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        
    Returns:
        numpy array: Combined feature matrix ready for clustering
    """
    X_parts = []
    
    # Add scaled numerical features
    if selected_numerical:
        for feat in selected_numerical:
            if feat in df_scaled.columns:
                X_parts.append(df_scaled[[feat]])
    
    # Add one-hot encoded categorical features
    if selected_categorical:
        # Define categorical columns to encode
        categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # One-hot encode categorical features
        df_encoded_temp = pd.get_dummies(df, columns=categorical_cols)
        
        # Add only the selected one-hot encoded categorical features
        for feat in selected_categorical:
            if feat in df_encoded_temp.columns:
                X_parts.append(df_encoded_temp[[feat]])
    
    # Combine all features
    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        # Fallback: use all numerical features
        X = df_scaled[selected_numerical] if selected_numerical else df_scaled
    
    # Ensure all values are numeric
    X = X.select_dtypes(include=[np.number])
    
    return X.values

def run_kmeans(df, df_scaled, all_selected_features, selected_numerical, selected_categorical, n_clusters):
    """
    Run K-Means clustering on selected features (both numerical and categorical) and create a 2D visualization.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        n_clusters: number of clusters for K-Means
        
    Returns:
        tuple: (plotly figure, cluster labels)
    """
    # Prepare features
    X = prepare_features_for_clustering(df, df_scaled, all_selected_features, selected_numerical, selected_categorical)
    
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Run PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA results and cluster labels
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = cluster_labels
    
    # Create scatter plot with clean, distinct colors
    fig = px.scatter(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        title=f'K-Means Clustering (k={n_clusters}) Visualization',
        labels={'PCA1': 'First Principal Component', 'PCA2': 'Second Principal Component'},
        color_discrete_sequence=['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8', '#fd7e14', '#e83e8c', '#20c997', '#6c757d']
    )
    
    # Update layout with clean styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='#333333'),
        title_font_size=18,
        title_font_color='#333333',
        xaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        yaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=12, color='#333333')
        )
    )
    
    return fig, cluster_labels

def run_dbscan(df, df_scaled, all_selected_features, selected_numerical, selected_categorical, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering on selected features (both numerical and categorical) and create a 2D visualization.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        tuple: (plotly figure, cluster_labels, n_clusters)
    """
    # Prepare features
    X = prepare_features_for_clustering(df, df_scaled, all_selected_features, selected_numerical, selected_categorical)
    
    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Run PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA results and cluster labels
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = cluster_labels
    
    # Create scatter plot
    fig = px.scatter(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        title=f'DBSCAN Clustering (eps={eps}, min_samples={min_samples}) - {n_clusters} clusters, {n_noise} noise points',
        labels={'PCA1': 'First Principal Component', 'PCA2': 'Second Principal Component'},
        color_discrete_sequence=['#6c757d'] + ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8', '#fd7e14', '#e83e8c', '#20c997']
    )
    
    # Update layout with clean styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='#333333'),
        title_font_size=16,
        title_font_color='#333333',
        xaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        yaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=12, color='#333333')
        )
    )
    
    return fig, cluster_labels, n_clusters

def run_hierarchical(df, df_scaled, all_selected_features, selected_numerical, selected_categorical, n_clusters, max_samples=5000):
    """
    Run Hierarchical (Agglomerative) Clustering on selected features (both numerical and categorical) and create a 2D visualization.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        n_clusters: number of clusters
        max_samples: maximum number of samples to use (for performance)
        
    Returns:
        tuple: (plotly figure, cluster_labels)
    """
    # Prepare features
    X = prepare_features_for_clustering(df, df_scaled, all_selected_features, selected_numerical, selected_categorical)
    
    # Sample data if too large (hierarchical clustering is O(n²) or O(n³))
    if len(X) > max_samples:
        # Use random sampling
        sample_indices = np.random.choice(len(X), size=max_samples, replace=False)
        X_sampled = X[sample_indices]
        df_sampled = df.iloc[sample_indices]
    else:
        X_sampled = X
        df_sampled = df
    
    # Run Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels_sampled = hierarchical.fit_predict(X_sampled)
    
    # Run PCA for 2D visualization (use sampled data for visualization)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_sampled)
    
    # Create DataFrame with PCA results and cluster labels
    # Use the labels from the sampled clustering for visualization
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = cluster_labels_sampled
    
    # Map labels back to full dataset if we sampled (for comparison metrics)
    if len(df) > max_samples:
        # Create full labels array with -1 for non-sampled points
        full_labels = np.full(len(df), -1)
        full_labels[sample_indices] = cluster_labels_sampled
        cluster_labels = full_labels
    else:
        cluster_labels = cluster_labels_sampled
    
    # Create scatter plot
    fig = px.scatter(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        title=f'Hierarchical Clustering (k={n_clusters}, linkage=ward)',
        labels={'PCA1': 'First Principal Component', 'PCA2': 'Second Principal Component'},
        color_discrete_sequence=['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8', '#fd7e14', '#e83e8c', '#20c997', '#6c757d']
    )
    
    # Update layout with clean styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='#333333'),
        title_font_size=16,
        title_font_color='#333333',
        xaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        yaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd', 
            title_font_size=14,
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=12, color='#333333')
        )
    )
    
    return fig, cluster_labels

def compare_clustering_algorithms(df, df_scaled, all_selected_features, selected_numerical, selected_categorical, kmeans_labels, dbscan_labels, hierarchical_labels):
    """
    Compare clustering algorithms using multiple metrics.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        kmeans_labels: cluster labels from K-Means
        dbscan_labels: cluster labels from DBSCAN
        hierarchical_labels: cluster labels from Hierarchical
        
    Returns:
        pandas DataFrame: Comparison metrics
    """
    # Prepare data
    X = prepare_features_for_clustering(df, df_scaled, all_selected_features, selected_numerical, selected_categorical)
    
    results = []
    
    # Evaluate each algorithm
    algorithms = [
        ('K-Means', kmeans_labels),
        ('DBSCAN', dbscan_labels),
        ('Hierarchical', hierarchical_labels)
    ]
    
    for name, labels in algorithms:
        # Filter out noise points (-1) for DBSCAN
        if -1 in labels:
            mask = labels != -1
            X_filtered = X[mask]
            labels_filtered = labels[mask]
        else:
            X_filtered = X
            labels_filtered = labels
        
        # Skip if no valid clusters
        n_clusters = len(set(labels_filtered))
        if n_clusters < 2 or len(X_filtered) < 2:
            results.append({
                'Algorithm': name,
                'Number of Clusters': n_clusters,
                'Silhouette Score': np.nan
            })
            continue
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X_filtered, labels_filtered)
        except:
            silhouette = np.nan
        
        results.append({
            'Algorithm': name,
            'Number of Clusters': n_clusters,
            'Silhouette Score': silhouette
        })
    
    return pd.DataFrame(results)
