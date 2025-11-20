import os

import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

TSNE_THRESHOLD = 0.05
TSNE_FEATURE_COUNT = 11
TSNE_FILENAME = os.path.join(os.path.dirname(__file__), '..', 'data', 'tsne_output.csv')

def run_pca(df, df_scaled, all_selected_features, selected_numerical, selected_categorical):
    """
    Run PCA on selected features (both numerical and categorical) and create a 2D visualization.
    
    Args:
        df: pandas DataFrame (original, unscaled)
        df_scaled: pandas DataFrame (with scaled numerical features)
        all_selected_features: list of all selected feature names (numerical + categorical)
        selected_numerical: list of selected numerical feature names
        selected_categorical: list of selected categorical feature names (one-hot encoded)
        
    Returns:
        plotly figure: Scatter plot of PCA components
    """
    # Prepare data for PCA
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
    
    # Run PCA with 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['readmitted'] = df['readmitted'].values
    
    # Create scatter plot with clean colors
    fig = px.scatter(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        color='readmitted',
        title='PCA Visualization of Patient Data',
        labels={'PCA1': 'First Principal Component', 'PCA2': 'Second Principal Component'},
        color_discrete_sequence=['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
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
    
    return fig


def load_precomputed_tsne(df):
    """
    Load precomputed t-SNE coordinates for the dataset when correlation threshold is 0.05.

    Args:
        df: Original dataframe used across the app (to align row order and labels)

    Returns:
        tuple: (tsne_df, error_message)
            - tsne_df: DataFrame containing t-SNE coordinates aligned with df (or None on failure)
            - error_message: None if successful, otherwise a user-friendly string
    """
    tsne_path = os.path.abspath(TSNE_FILENAME)
    if not os.path.exists(tsne_path):
        return None, "t-SNE data file is missing. Please regenerate tsne_output.csv."

    try:
        tsne_df = pd.read_csv(tsne_path)
    except Exception as exc:
        return None, f"Unable to read precomputed t-SNE data: {exc}"

    required_cols = {'tsne_dimension_1', 'tsne_dimension_2'}
    if not required_cols.issubset(tsne_df.columns):
        return None, "t-SNE file does not contain the expected coordinate columns."

    if len(tsne_df) != len(df):
        return None, "t-SNE data length does not match the current dataset. Please ensure tsne_output.csv was generated from the same data."

    tsne_df = tsne_df[['tsne_dimension_1', 'tsne_dimension_2']].copy()
    tsne_df.rename(
        columns={'tsne_dimension_1': 'TSNE1', 'tsne_dimension_2': 'TSNE2'},
        inplace=True
    )
    tsne_df['readmitted'] = df['readmitted'].values

    return tsne_df, None


def create_tsne_readmission_plot(tsne_df):
    """
    Build a Plotly scatter plot for precomputed t-SNE coordinates colored by readmission.

    Args:
        tsne_df: DataFrame returned by load_precomputed_tsne

    Returns:
        plotly figure
    """
    fig = px.scatter(
        tsne_df,
        x='TSNE1',
        y='TSNE2',
        color='readmitted',
        title='t-SNE Visualization of Patient Data (Threshold 0.05)',
        labels={'TSNE1': 't-SNE Dimension 1', 'TSNE2': 't-SNE Dimension 2'},
        color_discrete_sequence=['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
    )

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

    return fig


def create_tsne_cluster_plot(tsne_df, cluster_labels, algorithm_name):
    """
    Build a Plotly scatter plot for precomputed t-SNE coordinates colored by cluster labels.

    Args:
        tsne_df: DataFrame returned by load_precomputed_tsne
        cluster_labels: array-like, cluster assignments for each row in df
        algorithm_name: str, used in plot title

    Returns:
        plotly figure
    """
    if cluster_labels is None:
        raise ValueError("Cluster labels are required to build the t-SNE cluster plot.")

    if len(cluster_labels) != len(tsne_df):
        raise ValueError("Length mismatch between cluster labels and t-SNE data.")

    plot_df = tsne_df[['TSNE1', 'TSNE2']].copy()
    plot_df['Cluster'] = cluster_labels

    # Provide human-friendly labels for noise/unassigned points
    plot_df['Cluster'] = plot_df['Cluster'].astype(str)
    plot_df.loc[plot_df['Cluster'] == '-1', 'Cluster'] = 'Noise/Unassigned'

    fig = px.scatter(
        plot_df,
        x='TSNE1',
        y='TSNE2',
        color='Cluster',
        title=f'{algorithm_name} Clusters on t-SNE Projection',
        labels={'TSNE1': 't-SNE Dimension 1', 'TSNE2': 't-SNE Dimension 2'},
        color_discrete_sequence=['#6c757d', '#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8', '#fd7e14', '#e83e8c', '#20c997']
    )

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

    return fig
