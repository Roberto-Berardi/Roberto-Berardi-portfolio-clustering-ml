"""
Clustering module for portfolio clustering project.
Performs K-means and GMM clustering on stock features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_feature_matrix(all_features_dict, use_latest=True):
    """
    Prepare feature matrix from multiple stocks.
    
    Parameters:
    -----------
    all_features_dict : dict
        Dictionary with ticker as key and features DataFrame as value
    use_latest : bool
        If True, use only the most recent feature values for each stock
        If False, use all time periods (for time-series clustering)
    
    Returns:
    --------
    pd.DataFrame
        Feature matrix with stocks as rows
    """
    feature_data = []
    
    for ticker, features in all_features_dict.items():
        if len(features) == 0:
            continue
            
        if use_latest:
            # Use most recent values
            latest_features = features.iloc[-1].copy()
            latest_features['ticker'] = ticker
            feature_data.append(latest_features)
        else:
            # Use all time periods
            features_copy = features.copy()
            features_copy['ticker'] = ticker
            feature_data.append(features_copy)
    
    # Combine into single DataFrame
    if use_latest:
        feature_matrix = pd.DataFrame(feature_data)
    else:
        feature_matrix = pd.concat(feature_data, ignore_index=True)
    
    return feature_matrix


def standardize_features(feature_matrix, feature_cols):
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Feature matrix
    feature_cols : list
        List of column names to standardize
    
    Returns:
    --------
    np.ndarray
        Standardized features
    sklearn.preprocessing.StandardScaler
        Fitted scaler (for inverse transform if needed)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix[feature_cols])
    
    return X_scaled, scaler


def apply_pca(X_scaled, n_components=None, explained_variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Standardized features
    n_components : int or None
        Number of components (if None, use explained_variance_threshold)
    explained_variance_threshold : float
        Keep components that explain this much variance
    
    Returns:
    --------
    np.ndarray
        Transformed features
    sklearn.decomposition.PCA
        Fitted PCA model
    """
    if n_components is None:
        # Determine number of components based on explained variance
        pca = PCA()
        pca.fit(X_scaled)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
        
        print(f"  Selected {n_components} components explaining {cumsum[n_components-1]:.1%} of variance")
    
    # Fit PCA with selected components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca


def perform_kmeans(X, n_clusters=3, random_state=42):
    """
    Perform K-means clustering.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    sklearn.cluster.KMeans
        Fitted K-means model
    np.ndarray
        Cluster labels
    float
        Silhouette score
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score (requires at least 2 clusters and 2 samples per cluster)
    if n_clusters > 1 and len(X) > n_clusters:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = 0.0
    
    return kmeans, labels, silhouette


def perform_gmm(X, n_components=3, random_state=42):
    """
    Perform Gaussian Mixture Model clustering.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int
        Number of mixture components (clusters)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    sklearn.mixture.GaussianMixture
        Fitted GMM model
    np.ndarray
        Cluster labels
    float
        Silhouette score
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    
    # Calculate silhouette score
    if n_components > 1 and len(X) > n_components:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = 0.0
    
    return gmm, labels, silhouette


def label_clusters_by_volatility(feature_matrix, cluster_labels):
    """
    Label clusters as low/moderate/high volatility based on average volatility.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Feature matrix with 'volatility' column
    cluster_labels : np.ndarray
        Cluster assignments
    
    Returns:
    --------
    dict
        Mapping from cluster number to label (e.g., {0: 'low', 1: 'moderate', 2: 'high'})
    """
    # Calculate average volatility per cluster
    feature_matrix_copy = feature_matrix.copy()
    feature_matrix_copy['cluster'] = cluster_labels
    
    cluster_volatility = feature_matrix_copy.groupby('cluster')['volatility'].mean().sort_values()
    
    # Label clusters based on volatility ranking
    n_clusters = len(cluster_volatility)
    
    if n_clusters == 3:
        labels = {cluster_volatility.index[0]: 'low-volatility',
                 cluster_volatility.index[1]: 'moderate',
                 cluster_volatility.index[2]: 'high-volatility'}
    elif n_clusters == 2:
        labels = {cluster_volatility.index[0]: 'low-volatility',
                 cluster_volatility.index[1]: 'high-volatility'}
    else:
        # Generic labeling for other numbers of clusters
        labels = {cluster_volatility.index[i]: f'cluster-{i}' 
                 for i in range(n_clusters)}
    
    return labels


def analyze_clusters(feature_matrix, cluster_labels, cluster_names):
    """
    Analyze and summarize cluster characteristics.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
    cluster_names : dict
        Mapping from cluster number to name
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics for each cluster
    """
    feature_matrix_copy = feature_matrix.copy()
    feature_matrix_copy['cluster'] = cluster_labels
    feature_matrix_copy['cluster_name'] = feature_matrix_copy['cluster'].map(cluster_names)
    
    # Calculate summary statistics
    numeric_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
    summary = feature_matrix_copy.groupby('cluster_name')[numeric_cols].agg(['mean', 'std', 'count'])
    
    return summary


# Test function
if __name__ == "__main__":
    print("\n🧪 TESTING CLUSTERING MODULE\n")
    
    from data_loader import TEST_TICKERS, load_stock_data
    from feature_engineering import calculate_all_features
    import yfinance as yf
    
    # Load stock data
    print("Loading stock data...")
    stock_data = load_stock_data(TEST_TICKERS)
    
    # Download S&P 500
    print("Downloading S&P 500 data...")
    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)
    print(f"✓ Downloaded {len(sp500)} days of S&P 500 data\n")
    
    # Calculate features for all stocks
    print("Calculating features for all stocks...")
    all_features = {}
    for ticker in TEST_TICKERS:
        print(f"  Processing {ticker}...")
        features = calculate_all_features(stock_data[ticker], sp500, window=252)
        if len(features) > 0:
            all_features[ticker] = features
    
    print(f"\n✓ Calculated features for {len(all_features)} stocks\n")
    
    # Prepare feature matrix (use most recent values)
    print("Preparing feature matrix...")
    feature_matrix = prepare_feature_matrix(all_features, use_latest=True)
    print(f"✓ Feature matrix shape: {feature_matrix.shape}")
    print(f"  Stocks: {list(feature_matrix['ticker'])}\n")
    
    # Standardize features
    print("Standardizing features...")
    feature_cols = ['return', 'volatility', 'sharpe', 'max_drawdown', 'beta', 'correlation']
    X_scaled, scaler = standardize_features(feature_matrix, feature_cols)
    print(f"✓ Standardized {len(feature_cols)} features\n")
    
    # Apply PCA
    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled, n_components=None, explained_variance_threshold=0.95)
    print(f"✓ Reduced to {X_pca.shape[1]} principal components\n")
    
    # K-means clustering
    print("Performing K-means clustering (k=3)...")
    kmeans, kmeans_labels, kmeans_silhouette = perform_kmeans(X_pca, n_clusters=3, random_state=42)
    print(f"✓ K-means silhouette score: {kmeans_silhouette:.3f}\n")
    
    # GMM clustering
    print("Performing GMM clustering (k=3)...")
    gmm, gmm_labels, gmm_silhouette = perform_gmm(X_pca, n_components=3, random_state=42)
    print(f"✓ GMM silhouette score: {gmm_silhouette:.3f}\n")
    
    # Label clusters
    print("Labeling clusters by volatility...")
    kmeans_cluster_names = label_clusters_by_volatility(feature_matrix, kmeans_labels)
    gmm_cluster_names = label_clusters_by_volatility(feature_matrix, gmm_labels)
    
    # Add cluster assignments to feature matrix
    feature_matrix['kmeans_cluster'] = kmeans_labels
    feature_matrix['kmeans_label'] = feature_matrix['kmeans_cluster'].map(kmeans_cluster_names)
    feature_matrix['gmm_cluster'] = gmm_labels
    feature_matrix['gmm_label'] = feature_matrix['gmm_cluster'].map(gmm_cluster_names)
    
    print("\nK-Means Cluster Assignments:")
    print(feature_matrix[['ticker', 'volatility', 'kmeans_label']].sort_values('volatility'))
    
    print("\nGMM Cluster Assignments:")
    print(feature_matrix[['ticker', 'volatility', 'gmm_label']].sort_values('volatility'))
    
    # Analyze clusters
    print("\n" + "="*60)
    print("K-MEANS CLUSTER ANALYSIS")
    print("="*60)
    kmeans_summary = analyze_clusters(feature_matrix, kmeans_labels, kmeans_cluster_names)
    print(kmeans_summary)
    
    print("\n" + "="*60)
    print("GMM CLUSTER ANALYSIS")
    print("="*60)
    gmm_summary = analyze_clusters(feature_matrix, gmm_labels, gmm_cluster_names)
    print(gmm_summary)
    
    print("\n✓ Clustering test complete!")

def perform_clustering_old(X, method='kmeans', n_clusters=3, random_state=42):
    """
    Perform clustering using specified method.
    
    Args:
        X: Feature matrix (already scaled/PCA'd)
        method: 'kmeans' or 'gmm'
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        labels: Cluster labels for each sample
        score: Silhouette score
    """
    if method == 'kmeans':
        return perform_kmeans(X, n_clusters, random_state)
    elif method == 'gmm':
        return perform_gmm(X, n_clusters, random_state)
    else:
        raise ValueError(f"Unknown method: {method}")


# Fix perform_clustering to match expected return values
def perform_clustering(X, method='kmeans', n_clusters=3, random_state=42):
    """
    Perform clustering using specified method.
    Returns only labels and score (not the model object).
    """
    if method == 'kmeans':
        model, labels, score = perform_kmeans(X, n_clusters, random_state)
        return labels, score
    elif method == 'gmm':
        model, labels, score = perform_gmm(X, n_clusters, random_state)
        return labels, score
    else:
        raise ValueError(f"Unknown method: {method}")
