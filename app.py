import streamlit as st
import pandas as pd
from modules import data_processing, statistics, feature_selection, dimensionality, clustering, association

# Set page config with bright theme
st.set_page_config(
    page_title="Diabetic Patient Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean light theme
st.markdown("""
<style>
    /* Main app background - clean light */
    .main {
        background: #ffffff;
        padding: 1rem;
    }
    
    .stApp {
        background: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Header styling - beautiful gradient background */
    .main .block-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .main .block-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
    }
    
    /* Tabs styling - beautiful gradient tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        border-radius: 15px;
        padding: 6px;
        margin-bottom: 2rem;
        border: none;
        box-shadow: 0 5px 15px rgba(255, 154, 158, 0.3);
        overflow: hidden;
        position: relative;
    }
    
    .stTabs [data-baseweb="tab-list"]::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 20px;
        height: 100%;
        background: linear-gradient(90deg, transparent, #fecfef);
        pointer-events: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        margin: 3px;
        padding: 15px 25px;
        font-weight: 600;
        color: #333333;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 1);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
    }
    
    /* Data table styling - beautiful with gradients */
    .stDataFrame {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        border: none;
        position: relative;
    }
    
    .stDataFrame::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
    }
    
    .stDataFrame table {
        background: white !important;
        color: #333333 !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 15px !important;
        border: none !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stDataFrame td {
        background: white !important;
        color: #333333 !important;
        padding: 12px !important;
        border-bottom: 1px solid #f0f0f0 !important;
        transition: background-color 0.2s ease;
    }
    
    .stDataFrame tr:nth-child(even) {
        background: #f8f9fa !important;
    }
    
    .stDataFrame tr:hover {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
    }
    
    .stDataFrame tr:hover td {
        background: transparent !important;
    }
    
    /* Metrics styling - beautiful gradient cards */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: white !important;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric [data-testid="metric-label"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Plotly charts styling - beautiful with gradients */
    .js-plotly-plot {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .js-plotly-plot::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        z-index: 1;
    }
    
    /* Text styling - dark for readability */
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    
    p, div, span {
        color: #333333 !important;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #007bff;
    }
    
    /* Button styling - beautiful gradients */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ff5252, #d63031);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    
    /* Warning and info boxes - beautiful styling */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Slider styling - beautiful gradients */
    .stSlider > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Add beautiful section dividers */
    .stMarkdown hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #e0e0e0;
        color: #333333;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: white;
        border: 1px solid #e0e0e0;
        color: #333333;
    }
    
    /* Detailed Analysis section - ensure readable text */
    .stMarkdown [data-testid="stMarkdownContainer"] {
        background: white;
        color: #333333;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.title("🏥 Diabetic Patient Analytics Dashboard")
st.markdown("### Comprehensive Healthcare Data Analysis")

# Cache the data loading function
@st.cache_data
def load_data():
    """Load and clean the data"""
    return data_processing.load_and_clean_data()

# Load data
with st.spinner("🔄 Loading and processing data..."):
    df, df_scaled, numerical_features, id_mappings = load_data()


def ensure_tsne_cache(dataframe):
    """
    Load the precomputed t-SNE coordinates once per session and cache them.
    """
    if not st.session_state.get('tsne_cache_loaded', False):
        tsne_df, tsne_error = dimensionality.load_precomputed_tsne(dataframe)
        st.session_state['tsne_cache'] = tsne_df
        st.session_state['tsne_cache_error'] = tsne_error
        st.session_state['tsne_cache_loaded'] = True
    return st.session_state.get('tsne_cache'), st.session_state.get('tsne_cache_error')

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "Summary Statistics", 
    "Feature Selection",
    "Dimensionality Reduction", 
    "Patient Clustering", 
    "Association Rules"
])

# Tab 1: Overview
with tab1:
    st.header("📊 Project Overview")
    st.markdown("""
    - **Data Overview**: First 100 rows of the cleaned dataset
    - **Summary Statistics**: Descriptive statistics and distribution plots
    - **Feature Selection**: Correlation analysis to identify important features
    - **Dimensionality Reduction**: PCA visualization of patient data
    - **Patient Clustering**: K-Means clustering analysis with distinct colors
    - **Association Rules**: Apriori algorithm for finding disease patterns
    """)
    
    st.subheader("📋 Dataset Preview (First 100 rows)")
    # Remove admission_type, discharge_disposition, admission_source columns from preview
    preview_df = df.head(100).copy()
    columns_to_hide = ['admission_type', 'discharge_disposition', 'admission_source']
    preview_df = preview_df.drop(columns=[col for col in columns_to_hide if col in preview_df.columns])
    st.dataframe(preview_df)
    
    st.subheader("📈 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numerical Features", len(numerical_features))
    with col4:
        st.metric("Readmission Rate", f"{(df['readmitted_bool'].mean() * 100):.1f}%")

# Tab 2: Summary Statistics
with tab2:
    st.header("📊 Summary Statistics")
    
    st.subheader("📈 Descriptive Statistics (Before Scaling)")
    summary_stats = statistics.get_summary_statistics(df, numerical_features)
    st.dataframe(summary_stats)
    st.caption("**Note:** The following features have been capped to handle outliers: number_emergency (capped at 15), number_outpatient (capped at 15), num_medications (capped at 50)")
    
    st.subheader("📈 Descriptive Statistics (After Scaling)")
    summary_stats_scaled = statistics.get_summary_statistics(df_scaled, numerical_features)
    st.dataframe(summary_stats_scaled)
    st.caption("**Note:** Features have been standardized (mean=0, std=1) using StandardScaler after clipping outliers.")
    
    st.subheader("📊 Data Distributions")
    hist_fig, pie_fig = statistics.plot_distributions(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hist_fig, use_container_width=True)
    with col2:
        st.plotly_chart(pie_fig, use_container_width=True)

# Tab 3: Feature Selection
with tab3:
    st.header("🎯 Feature Selection")
    st.markdown("""
    This page analyzes which features are most correlated with **readmission**.
    
    We create a binary target (1 for readmitted, 0 for not) and then calculate the correlation
    of all our numerical features and key categorical features (after one-hot encoding them).
    
    **Note:** Numerical features have been standardized (scaled) using StandardScaler after clipping outliers,
    ensuring that correlation analysis is not biased by different feature scales.
    """)
    
    # Get the correlation data and figure from our new module (using scaled data)
    corr_df, fig, corr_matrix, all_candidate_features = feature_selection.get_feature_correlations(df_scaled, numerical_features)
    
    if not corr_df.empty:
        # Add a slider to filter
        threshold = st.slider("Select Correlation Threshold", 0.0, 0.2, 0.05, 0.01)
        
        # Get features above threshold (both numerical and categorical)
        features_above_threshold = corr_df[corr_df['Correlation'] > threshold].index.tolist()
        # Filter to only include original numerical features (for backward compatibility)
        selected_features = [f for f in features_above_threshold if f in numerical_features]
        
        # Store ALL selected features (both numerical and categorical) in session state for use in PCA
        st.session_state['selected_features'] = features_above_threshold  # All features (numerical + categorical)
        st.session_state['selected_numerical_features'] = selected_features  # Only numerical (for backward compatibility)
        st.session_state['correlation_threshold'] = threshold
        
        # Display correlation heatmap with explanation - showing ALL candidate features
        st.subheader("📊 Correlation Heatmap (All Candidate Features)")
        st.markdown("""
        **Understanding the Heatmap:**
        - **Red colors** indicate positive correlation (features that increase together)
        - **Blue colors** indicate negative correlation (features that move in opposite directions)
        - **Darker colors** mean stronger correlations
        - The diagonal is always 1.0 (each feature is perfectly correlated with itself)
        - **Use this heatmap to select features:** Look for features with high correlation to readmission and low correlation with each other (to avoid redundancy)
        """)
        heatmap_fig = feature_selection.create_correlation_heatmap(corr_matrix, all_candidate_features, corr_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Display the main bar chart
        st.subheader("📈 Feature Correlation with Readmission")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the features above the threshold
        st.subheader("Features with Correlation Above Threshold")
        features_df = corr_df[corr_df['Correlation'] > threshold]
        st.dataframe(features_df)
        
        # CSV download for selected features
        if selected_features or len(features_above_threshold) > 0:
            # Get all selected features (both numerical and categorical one-hot encoded)
            all_selected_features = features_above_threshold.copy()
            
            # Separate into numerical and categorical (one-hot encoded)
            selected_numerical = [f for f in all_selected_features if f in numerical_features]
            selected_categorical_encoded = [f for f in all_selected_features if f not in numerical_features]
            
            # Start building the final dataframe
            selected_data = pd.DataFrame(index=df.index)
            
            # Add scaled numerical features
            if selected_numerical:
                for feat in selected_numerical:
                    if feat in df_scaled.columns:
                        selected_data[feat] = df_scaled[feat]
            
            # Add one-hot encoded categorical features
            if selected_categorical_encoded:
                # Define categorical columns to encode (same as in feature_selection.py)
                categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
                categorical_cols = [col for col in categorical_cols if col in df.columns]
                
                # One-hot encode categorical features
                df_encoded_temp = pd.get_dummies(df, columns=categorical_cols)
                
                # Add only the selected one-hot encoded categorical features
                for feat in selected_categorical_encoded:
                    if feat in df_encoded_temp.columns:
                        selected_data[feat] = df_encoded_temp[feat]
            
            # Convert to CSV
            csv_data = selected_data.to_csv(index=False)
            
            # Count total features
            total_features = len(selected_numerical) + len(selected_categorical_encoded)
            
            # Add download button
            st.download_button(
                label="📥 Download Selected Features CSV (Scaled)",
                data=csv_data,
                file_name=f"selected_features_threshold_{threshold:.2f}.csv",
                mime="text/csv",
                help="Download the dataset with scaled numerical features and one-hot encoded categorical features for offline analysis (e.g., t-SNE)"
            )
            st.caption(f"**Note:** This CSV contains {len(selected_data)} rows and {total_features} selected features ({len(selected_numerical)} scaled numerical + {len(selected_categorical_encoded)} one-hot encoded categorical). Use this for offline analysis to avoid freezing Streamlit during computationally intensive operations like t-SNE.")
        
        # Show which features will be used in PCA and Clustering
        if selected_features or len(features_above_threshold) > 0:
            # Get all selected features (both numerical and categorical)
            all_selected_features = features_above_threshold.copy()
            selected_numerical = [f for f in all_selected_features if f in numerical_features]
            selected_categorical = [f for f in all_selected_features if f not in numerical_features]
            
            total_features = len(all_selected_features)
            feature_list = ', '.join(all_selected_features[:20])  # Show first 20
            if len(all_selected_features) > 20:
                feature_list += f", ... and {len(all_selected_features) - 20} more"
            
            st.info(f"**{total_features} features** will be used in PCA and Clustering ({len(selected_numerical)} numerical + {len(selected_categorical)} categorical): {feature_list}")
            st.markdown(f"""
            **Why these features?**
            - These features have correlation > {threshold:.2f} with readmission
            - They are the most predictive numerical and categorical features
            - Using only these features reduces noise and improves model performance
            """)
        else:
            st.warning("No numerical features above the threshold. Please lower the threshold or use default features.")
    
    else:
        st.error("Could not calculate feature correlations.")

# Tab 4: Dimensionality Reduction
with tab4:
    st.header("🔍 Dimensionality Reduction (PCA)")
    st.markdown("Principal Component Analysis to visualize high-dimensional data in 2D")

    threshold_used = st.session_state.get('correlation_threshold')
    
    # Use selected features from Feature Selection page, or fallback to all numerical features
    if 'selected_features' in st.session_state and len(st.session_state['selected_features']) > 0:
        all_selected_features = st.session_state['selected_features']
        display_threshold = threshold_used if threshold_used is not None else dimensionality.TSNE_THRESHOLD
        
        # Separate numerical and categorical features
        selected_numerical = [f for f in all_selected_features if f in numerical_features]
        selected_categorical = [f for f in all_selected_features if f not in numerical_features]
        
        st.info(f"Using **{len(all_selected_features)} features** selected from Feature Selection page ({len(selected_numerical)} numerical + {len(selected_categorical)} categorical, threshold: {display_threshold:.2f})")
        
        # Prepare data for PCA: scaled numerical + one-hot encoded categorical
        pca_fig = dimensionality.run_pca(df, df_scaled, all_selected_features, selected_numerical, selected_categorical)
    else:
        features_to_use = numerical_features
        st.info(f"Using all **{len(features_to_use)} numerical features**. Visit the Feature Selection page to filter by correlation threshold.")
        pca_fig = dimensionality.run_pca(df, df_scaled, features_to_use, features_to_use, [])
        threshold_used = None
    
    st.plotly_chart(pca_fig, use_container_width=True)
    
    if 'selected_features' in st.session_state and len(st.session_state['selected_features']) > 0:
        num_features = len(st.session_state['selected_features'])
    else:
        num_features = len(numerical_features)
    
    st.markdown(f"""
    **Interpretation:**
    - Each point represents a patient
    - Colors indicate readmission status
    - PCA reduces the {num_features} selected features to 2 principal components
    """)

    st.subheader("🌀 t-SNE Visualization")
    tsne_threshold_active = (
        threshold_used is not None and abs(threshold_used - dimensionality.TSNE_THRESHOLD) < 1e-6
    )
    if tsne_threshold_active:
        tsne_df, tsne_error = ensure_tsne_cache(df)
        if tsne_df is not None:
            tsne_fig = dimensionality.create_tsne_readmission_plot(tsne_df)
            st.plotly_chart(tsne_fig, use_container_width=True)
            st.caption("Precomputed t-SNE (threshold 0.05, 11 selected features).")
        else:
            st.error(tsne_error or "Unable to display the precomputed t-SNE visualization.")
    else:
        st.info(
            "t-SNE is computationally expensive. Precomputed results are available only when the correlation "
            "threshold is set to 0.05 (11 features). Adjust the slider in Feature Selection to 0.05 to view the plot."
        )

# Tab 5: Patient Clustering
with tab5:
    st.header("🎯 Patient Clustering - Algorithm Comparison")
    st.markdown("Compare three clustering algorithms: K-Means, DBSCAN, and Hierarchical Clustering")

    threshold_used = st.session_state.get('correlation_threshold')
    
    # Use selected features from Feature Selection page, or fallback to all numerical features
    if 'selected_features' in st.session_state and len(st.session_state['selected_features']) > 0:
        all_selected_features = st.session_state['selected_features']
        display_threshold = threshold_used if threshold_used is not None else dimensionality.TSNE_THRESHOLD
        
        # Separate numerical and categorical features
        selected_numerical = [f for f in all_selected_features if f in numerical_features]
        selected_categorical = [f for f in all_selected_features if f not in numerical_features]
        
        st.info(f"Using **{len(all_selected_features)} features** selected from Feature Selection page ({len(selected_numerical)} numerical + {len(selected_categorical)} categorical, threshold: {display_threshold:.2f})")
        
        # Store for clustering functions
        features_to_use = all_selected_features
        features_to_use_numerical = selected_numerical
        features_to_use_categorical = selected_categorical
    else:
        features_to_use = numerical_features
        features_to_use_numerical = numerical_features
        features_to_use_categorical = []
        st.info(f"Using all **{len(features_to_use)} numerical features**. Visit the Feature Selection page to filter by correlation threshold.")
        threshold_used = None

    tsne_cluster_available = False
    tsne_cluster_df = None
    tsne_status_msg = ""
    if threshold_used is not None and abs(threshold_used - dimensionality.TSNE_THRESHOLD) < 1e-6:
        tsne_cluster_df, tsne_cluster_error = ensure_tsne_cache(df)
        if tsne_cluster_df is not None:
            tsne_cluster_available = True
        else:
            tsne_status_msg = tsne_cluster_error or "Unable to load precomputed t-SNE data."
    else:
        tsne_status_msg = "t-SNE cluster overlays are precomputed only when the correlation threshold is set to 0.05 (11 selected features)."
    
    if not tsne_cluster_available and tsne_status_msg:
        st.info(tsne_status_msg)
    
    # Parameters section
    st.subheader("⚙️ Algorithm Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("K-Means: Number of Clusters", min_value=2, max_value=10, value=3)
    
    with col2:
        eps = st.slider("DBSCAN: Epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("DBSCAN: Min Samples", min_value=3, max_value=20, value=5)
    
    with col3:
        hierarchical_n_clusters = st.slider("Hierarchical: Number of Clusters (k)", min_value=2, max_value=10, value=3)
    
    # Run all three clustering algorithms with error handling
    try:
        with st.spinner("🔄 Running K-Means clustering..."):
            kmeans_fig, kmeans_labels = clustering.run_kmeans(df, df_scaled, features_to_use, features_to_use_numerical, features_to_use_categorical, n_clusters)
    except Exception as e:
        st.error(f"Error running K-Means: {str(e)}")
        kmeans_fig = None
        kmeans_labels = None
    
    try:
        with st.spinner("🔄 Running DBSCAN clustering..."):
            dbscan_fig, dbscan_labels, dbscan_n_clusters = clustering.run_dbscan(df, df_scaled, features_to_use, features_to_use_numerical, features_to_use_categorical, eps, min_samples)
    except Exception as e:
        st.error(f"Error running DBSCAN: {str(e)}")
        dbscan_fig = None
        dbscan_labels = None
    
    try:
        with st.spinner("🔄 Running Hierarchical clustering (this may take a while for large datasets)..."):
            hierarchical_fig, hierarchical_labels = clustering.run_hierarchical(df, df_scaled, features_to_use, features_to_use_numerical, features_to_use_categorical, hierarchical_n_clusters)
    except Exception as e:
        st.error(f"Error running Hierarchical clustering: {str(e)}")
        st.warning("⚠️ Hierarchical clustering may be too slow for large datasets. Consider using a smaller sample or different algorithm.")
        hierarchical_fig = None
        hierarchical_labels = None
    
    # Display all three visualizations side by side
    st.subheader("📊 Clustering Visualizations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if kmeans_fig is not None:
            st.plotly_chart(kmeans_fig, use_container_width=True)
            st.caption("**K-Means**: Partition-based clustering with predefined number of clusters")
            if tsne_cluster_available and kmeans_labels is not None:
                kmeans_tsne_fig = dimensionality.create_tsne_cluster_plot(
                    tsne_cluster_df, kmeans_labels, f'K-Means (k={n_clusters})'
                )
                st.plotly_chart(kmeans_tsne_fig, use_container_width=True)
                st.caption("t-SNE projection of K-Means clusters (threshold 0.05).")
        else:
            st.error("K-Means visualization not available")
    
    with col2:
        if dbscan_fig is not None:
            st.plotly_chart(dbscan_fig, use_container_width=True)
            st.caption("**DBSCAN**: Density-based clustering that finds clusters of varying shapes and identifies noise")
            if tsne_cluster_available and dbscan_labels is not None:
                dbscan_tsne_fig = dimensionality.create_tsne_cluster_plot(
                    tsne_cluster_df, dbscan_labels, f'DBSCAN (eps={eps}, min_samples={min_samples})'
                )
                st.plotly_chart(dbscan_tsne_fig, use_container_width=True)
                st.caption("t-SNE projection of DBSCAN clusters (threshold 0.05).")
        else:
            st.error("DBSCAN visualization not available")
    
    with col3:
        if hierarchical_fig is not None:
            st.plotly_chart(hierarchical_fig, use_container_width=True)
            st.caption("**Hierarchical**: Builds a tree of clusters using Ward linkage (sampled for performance)")
            if tsne_cluster_available and hierarchical_labels is not None:
                hierarchical_tsne_fig = dimensionality.create_tsne_cluster_plot(
                    tsne_cluster_df, hierarchical_labels, f'Hierarchical (k={hierarchical_n_clusters})'
                )
                st.plotly_chart(hierarchical_tsne_fig, use_container_width=True)
                st.caption("t-SNE projection of Hierarchical clusters (threshold 0.05).")
        else:
            st.error("Hierarchical clustering visualization not available")
    
    # Model Comparison Section
    st.subheader("📈 Algorithm Comparison & Analysis")
    
    # Only compare algorithms that ran successfully
    if kmeans_labels is not None and dbscan_labels is not None and hierarchical_labels is not None:
        # Calculate comparison metrics
        comparison_df = clustering.compare_clustering_algorithms(
            df, df_scaled, features_to_use, features_to_use_numerical, features_to_use_categorical, kmeans_labels, dbscan_labels, hierarchical_labels
        )
    else:
        st.warning("⚠️ Cannot compare algorithms - some failed to run. Please check the error messages above.")
        comparison_df = pd.DataFrame()
    
    # Display comparison table (Silhouette-focused)
    st.dataframe(
        comparison_df[['Algorithm', 'Number of Clusters', 'Silhouette Score']],
        use_container_width=True
    )
    
    # Find best algorithm based on metrics
    valid_df = comparison_df.dropna(subset=['Silhouette Score'])
    
    if len(valid_df) > 0:
        # Best silhouette score (higher is better)
        best_silhouette = valid_df.loc[valid_df['Silhouette Score'].idxmax()]
        
        # Display recommendations (Silhouette-driven)
        st.markdown("### 🏆 Algorithm Recommendations")
        
        st.metric(
            "Best Silhouette Score",
            f"{best_silhouette['Algorithm']}",
            f"{best_silhouette['Silhouette Score']:.3f}"
        )
        
        # Detailed analysis
        st.markdown("### 📝 Detailed Analysis")
        
        # Use a container with white background for better readability
        with st.container():
            st.markdown(f"""
**Overall Best Algorithm: {best_silhouette['Algorithm']}**

**Why {best_silhouette['Algorithm']} performs best:**
            """)
        
            if best_silhouette['Algorithm'] == 'K-Means':
                st.markdown(f"""
- **Silhouette Score: {best_silhouette['Silhouette Score']:.3f}** - Indicates well-separated, cohesive clusters
- **Advantages:**
  - Fast and efficient for large datasets
  - Works well with spherical clusters
  - Produces consistent results with fixed number of clusters
- **Best for:** When you know the number of clusters and want fast, interpretable results
                """)
            
            elif best_silhouette['Algorithm'] == 'DBSCAN':
                st.markdown(f"""
- **Silhouette Score: {best_silhouette['Silhouette Score']:.3f}** - Good cluster separation despite noise handling
- **Advantages:**
  - Automatically determines number of clusters
  - Handles noise and outliers effectively
  - Can find clusters of arbitrary shapes
- **Best for:** When you don't know the number of clusters and want to identify outliers
                """)
            
            else:  # Hierarchical
                st.markdown(f"""
- **Silhouette Score: {best_silhouette['Silhouette Score']:.3f}** - Good cluster quality with hierarchical structure
- **Advantages:**
  - Creates interpretable cluster hierarchy
  - No need to specify number of clusters upfront (can cut tree at different levels)
  - Works well with Ward linkage for compact clusters
- **Best for:** When you want to understand cluster relationships and hierarchy
                """)
            
            st.markdown("""
**Silhouette Score Explanation:**
- **Range**: -1 to 1, where higher is better
- **Measures**: How similar objects are to their own cluster vs other clusters
- **Interpretation**: Values close to 1 indicate well-separated, cohesive clusters
            """)
    
    else:
        st.warning("⚠️ Unable to calculate comparison metrics. Some algorithms may have produced invalid clusters.")
    
    # Cluster Profile Section - Show after best algorithm is selected
    if len(valid_df) > 0:
        best_algorithm = best_silhouette['Algorithm']
        st.markdown("---")
        st.subheader("📊 Cluster Profile")
        
        # Determine which labels to use based on best algorithm
        if best_algorithm == 'K-Means' and kmeans_labels is not None:
            cluster_labels_to_use = kmeans_labels
            df_for_profile = df.copy()
            df_for_profile['Cluster'] = cluster_labels_to_use
        elif best_algorithm == 'DBSCAN' and dbscan_labels is not None:
            cluster_labels_to_use = dbscan_labels
            # Filter out noise points for profile
            mask = cluster_labels_to_use != -1
            df_for_profile = df[mask].copy()
            df_for_profile['Cluster'] = cluster_labels_to_use[mask]
        elif best_algorithm == 'Hierarchical' and hierarchical_labels is not None:
            cluster_labels_to_use = hierarchical_labels
            # Filter out -1 labels if any (from sampling)
            mask = cluster_labels_to_use != -1
            df_for_profile = df[mask].copy()
            df_for_profile['Cluster'] = cluster_labels_to_use[mask]
        else:
            cluster_labels_to_use = None
            df_for_profile = None
        
        if df_for_profile is not None and 'Cluster' in df_for_profile.columns:
            # Create profile table
            profile_data = []
            
            # Get numerical features averages by cluster
            for cluster_id in sorted(df_for_profile['Cluster'].unique()):
                cluster_data = df_for_profile[df_for_profile['Cluster'] == cluster_id].copy()
                row_data = {'Cluster': f'Cluster {cluster_id}'}
                
                # Add averages for numerical features
                for feat in numerical_features:
                    if feat in cluster_data.columns:
                        row_data[feat] = round(cluster_data[feat].mean(), 2)
                
                # Add counts for each categorical feature
                categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
                for cat_col in categorical_cols:
                    if cat_col in cluster_data.columns:
                        # Get counts for each category value in this cluster
                        value_counts = cluster_data[cat_col].value_counts()
                        if len(value_counts) > 0:
                            # Format as "Category1 (count1), Category2 (count2), ..."
                            count_str = ", ".join([f"{val} ({count})" for val, count in value_counts.items()])
                            row_data[cat_col] = count_str
                        else:
                            row_data[cat_col] = 'N/A'

                # Add readmission distribution
                readmitted_col = None
                if 'readmitted' in cluster_data.columns:
                    readmitted_col = 'readmitted'
                elif 'readmitted_bool' in cluster_data.columns:
                    readmitted_col = 'readmitted_bool'

                if readmitted_col:
                    readmit_counts = cluster_data[readmitted_col].value_counts()
                    if len(readmit_counts) > 0:
                        readmit_str = ", ".join([f"{val}: {count}" for val, count in readmit_counts.items()])
                        row_data['Readmission'] = readmit_str
                    else:
                        row_data['Readmission'] = 'N/A'
                else:
                    row_data['Readmission'] = 'N/A'
                
                profile_data.append(row_data)
            
            profile_df = pd.DataFrame(profile_data)
            
            if not profile_df.empty:
                st.dataframe(profile_df, use_container_width=True)
            else:
                st.info("Cluster profile data is being calculated...")
        else:
            st.info("Cluster profile data is not available.")

# Tab 6: Association Rules
with tab6:
    st.header("🔗 Association Rules (Apriori)")
    st.markdown("Discover patterns in patient diagnoses using association rule mining")
    
    with st.spinner("🔍 Mining association rules..."):
        rules_df = association.find_association_rules(df)
    
    if not rules_df.empty:
        st.subheader("📋 Discovered Disease Association Rules")
        
        # Show top 10 rules - only keep required columns
        st.write("**Top 10 Most Significant Rules:**")
        top_rules = rules_df.head(10)
        
        # Select only the required columns
        required_columns = ['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift']
        # Check which columns exist (column names might vary)
        available_columns = []
        for col in required_columns:
            if col in top_rules.columns:
                available_columns.append(col)
            # Try with underscores
            elif col.replace(' ', '_') in top_rules.columns:
                available_columns.append(col.replace(' ', '_'))
        
        if available_columns:
            top_rules_display = top_rules[available_columns].copy()
            # Rename columns to match expected format
            column_mapping = {
                'antecedent support': 'antecedents support',
                'antecedent_support': 'antecedents support',
                'consequent support': 'consequents support',
                'consequent_support': 'consequents support'
            }
            top_rules_display = top_rules_display.rename(columns=column_mapping)
            st.dataframe(top_rules_display, use_container_width=True)
        else:
            st.dataframe(top_rules, use_container_width=True)
        
        st.subheader("📊 Rule Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules", len(rules_df))
        with col2:
            if len(rules_df) > 0:
                st.metric("Highest Lift", f"{rules_df['lift'].max():.2f}")
        with col3:
            if len(rules_df) > 0:
                st.metric("Average Confidence", f"{rules_df['confidence'].mean():.2f}")
        with col4:
            if len(rules_df) > 0:
                st.metric("Average Support", f"{rules_df['support'].mean():.3f}")
        
        st.markdown("""
        **Rule Interpretation:**
        - **Antecedents**: Disease conditions that appear together
        - **Consequents**: Disease conditions that are likely to follow
        - **Support**: How frequently the rule appears in the dataset (0-1)
        - **Confidence**: How often the consequent appears with the antecedent (0-1)
        - **Lift**: How much more likely the consequent is given the antecedent (>1 means positive association)
        
        **Example**: If a rule shows "Diabetes mellitus → Hypertensive chronic kidney disease" with lift 2.5, 
        it means patients with diabetes are 2.5 times more likely to also have hypertensive kidney disease.
        """)
        
        # Show some example rules in a more readable format
        if len(rules_df) > 0:
            st.subheader("🎯 Key Insights")
            for i, (_, rule) in enumerate(rules_df.head(3).iterrows()):
                st.write(f"**Rule {i+1}:** If patient has **{rule['antecedents']}**, then they are {rule['lift']:.1f}x more likely to have **{rule['consequents']}** (Confidence: {rule['confidence']:.1%})")
    else:
        st.warning("⚠️ No association rules found with the current parameters. This might be due to:")
        st.write("• Low minimum support threshold")
        st.write("• Insufficient data patterns")
        st.write("• Need to adjust algorithm parameters")

