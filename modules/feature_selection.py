import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_feature_correlations(df, numerical_features):
    """
    Calculates the correlation of all features with the readmission target.
    Note: The dataframe should already be scaled before calling this function.
    
    Args:
        df (pd.DataFrame): The main dataframe (should be pre-scaled).
        numerical_features (list): The list of 8 numerical feature names.
        
    Returns:
        tuple: (correlation_dataframe, plotly_bar_chart)
    """
    
    df_fs = df.copy()
    
    # 1. Create the binary target variable for correlation
    df_fs['readmitted_target'] = df_fs['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
    
    # 2. Define key categorical features to encode (using ID columns)
    categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    # Filter to only include columns that actually exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df_fs.columns]
    
    # 3. One-hot encode these features
    df_encoded = pd.get_dummies(df_fs, columns=categorical_cols)
    
    # 4. Create a final list of all features to check
    all_features = [f for f in numerical_features if f in df_encoded.columns]
    
    # Get encoded categorical columns (they will have format like 'race_Caucasian', 'gender_Male', etc.)
    encoded_categorical_cols = []
    for cat_col in categorical_cols:
        prefix = cat_col + '_'
        encoded_categorical_cols.extend([col for col in df_encoded.columns if col.startswith(prefix)])
    
    all_features.extend(encoded_categorical_cols)
    
    # 5. Select only numeric columns for correlation (including the target)
    # Get numeric columns
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    # Also include one-hot encoded columns (they should be numeric but let's be explicit)
    # and make sure readmitted_target is included
    cols_for_corr = numeric_cols.copy()
    
    # Add encoded categorical columns if they're not already numeric
    # Note: get_dummies creates numeric columns, so they should already be included
    # But we'll add them explicitly to be safe
    for col in encoded_categorical_cols:
        if col not in cols_for_corr and col in df_encoded.columns:
            cols_for_corr.append(col)
    
    # Make sure readmitted_target is included
    if 'readmitted_target' not in cols_for_corr and 'readmitted_target' in df_encoded.columns:
        cols_for_corr.append('readmitted_target')
    
    # Also ensure all numerical_features are included
    for feat in numerical_features:
        if feat not in cols_for_corr and feat in df_encoded.columns:
            cols_for_corr.append(feat)
    
    # Filter to only numeric columns
    df_numeric = df_encoded[cols_for_corr]
    
    # 6. Calculate the full correlation matrix
    try:
        corr_matrix = df_numeric.corr()
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        # Return empty on failure
        return pd.DataFrame(), px.bar(), pd.DataFrame(), []

    # 7. Get correlation with the target, take absolute value, and sort
    if 'readmitted_target' in corr_matrix.columns:
        corr_target = corr_matrix['readmitted_target'].abs().sort_values(ascending=False)
        
        # 8. Keep only the features we're interested in (not the target itself)
        corr_target = corr_target.drop('readmitted_target', errors='ignore')
        
        # Filter to only features that exist in our all_features list
        corr_target_filtered = corr_target[corr_target.index.isin(all_features)]
        
        # If filtering removed everything, at least keep the numerical features
        if len(corr_target_filtered) == 0:
            # Fallback: just use numerical features that exist in correlation matrix
            available_numerical = [f for f in numerical_features if f in corr_target.index]
            if len(available_numerical) > 0:
                corr_target_filtered = corr_target[corr_target.index.isin(available_numerical)]
            else:
                # Last resort: return top correlations regardless
                corr_target_filtered = corr_target.head(20)
        
        corr_target = corr_target_filtered
        
        # Check if we have any correlations
        if len(corr_target) == 0:
            return pd.DataFrame(), px.bar(), pd.DataFrame(), all_features
        
        # 9. Create a Plotly bar chart with readable text
        corr_df_plot = corr_target.to_frame(name='Correlation').reset_index()
        fig = px.bar(
            corr_df_plot,
            x='Correlation', 
            y='index', 
            orientation='h',
            title='Feature Correlation with Readmission',
            labels={'Correlation': 'Absolute Correlation', 'index': 'Feature'},
            text='Correlation'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside', textfont=dict(size=12))
        fig.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            yaxis=dict(
                autorange="reversed", 
                tickfont=dict(size=14), 
                title=dict(font=dict(size=16))
            ), # Show highest on top
            xaxis=dict(
                tickfont=dict(size=14), 
                title=dict(font=dict(size=16))
            ),
            title=dict(font=dict(size=20)),
            height=max(600, len(corr_df_plot) * 30)
        )
        
        return corr_target.to_frame(name='Correlation'), fig, corr_matrix, all_features
    else:
        # Return empty on failure
        return pd.DataFrame(), px.bar(), pd.DataFrame(), []

def create_correlation_heatmap(corr_matrix, all_candidate_features=None, corr_df=None):
    """
    Create a correlation heatmap for ALL candidate features (before selection).
    
    Args:
        corr_matrix (pd.DataFrame): Full correlation matrix
        all_candidate_features (list): List of ALL candidate features to show in heatmap
        corr_df (pd.DataFrame): DataFrame with correlation values for explanation
        
    Returns:
        plotly figure: Heatmap figure
    """
    if corr_matrix.empty:
        return px.bar()
    
    # Show ALL candidate features (not just selected ones)
    # This helps users understand correlations before selecting features
    if all_candidate_features is not None and len(all_candidate_features) > 0:
        # Filter to only include candidate features that exist in the matrix
        available_features = [f for f in all_candidate_features if f in corr_matrix.index and f in corr_matrix.columns]
        if len(available_features) > 0:
            corr_subset = corr_matrix.loc[available_features, available_features]
        else:
            # Fallback: use top features by correlation with target
            if 'readmitted_target' in corr_matrix.columns:
                top_features = corr_matrix['readmitted_target'].abs().sort_values(ascending=False).head(30).index.tolist()
                top_features = [f for f in top_features if f != 'readmitted_target']
                corr_subset = corr_matrix.loc[top_features, top_features]
            else:
                corr_subset = corr_matrix.iloc[:30, :30]
    else:
        # Use top features by default
        if 'readmitted_target' in corr_matrix.columns:
            top_features = corr_matrix['readmitted_target'].abs().sort_values(ascending=False).head(30).index.tolist()
            top_features = [f for f in top_features if f != 'readmitted_target']
            corr_subset = corr_matrix.loc[top_features, top_features]
        else:
            corr_subset = corr_matrix.iloc[:30, :30]
    
    # Calculate dynamic height based on number of features
    n_features = len(corr_subset)
    height = max(600, n_features * 40)
    width = max(800, n_features * 40)
    
    # Create heatmap with reversed colorscale: Blue for positive, Red for negative
    # Using 'RdBu_r' reverses the default (or we can use a custom colorscale)
    fig = go.Figure(data=go.Heatmap(
        z=corr_subset.values,
        x=corr_subset.columns,
        y=corr_subset.index,
        colorscale='RdBu_r',  # Reversed: Blue for positive, Red for negative
        zmid=0,
        text=corr_subset.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        colorbar=dict(title=dict(text="Correlation", font=dict(size=14)))
    ))
    
    fig.update_layout(
        title=dict(text='Feature Correlation Heatmap (All Candidate Features)', font=dict(size=20)),
        xaxis=dict(
            title=dict(text='Features', font=dict(size=16)), 
            tickfont=dict(size=12, color='#000000')
        ),
        yaxis=dict(
            title=dict(text='Features', font=dict(size=16)), 
            tickfont=dict(size=12, color='#000000')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        width=width
    )
    
    return fig

