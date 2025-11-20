🏥 Diabetic Patient Comorbidity & Readmission Dashboard

A Streamlit-based interactive analytics dashboard for exploring comorbidities, clustering patterns, and readmission risk among diabetic hospital patients.
This project provides a complete end-to-end workflow — from data cleaning to advanced analytics and visual insights.

Live Demo (HuggingFace Spaces):
https://huggingface.co/spaces/Megha-Harsha/Hospital-Patient-Comorbidity-and-Readmission-Dashboard
Please wait a few minutes(~10 mins) after clicking — the app processes the dataset on startup.

Key Functionalities

  -Data Cleaning & Preprocessing:
  Handles missing values, caps outliers, standardizes numerical features, and prepares encoded categorical features.
  
  -Summary Statistics:
  Descriptive statistics, distribution plots, pie charts, and data insights.
  
  -Feature Selection:
  Correlation heatmaps, threshold-based selection, and downloadable selected feature sets.
  
  -Dimensionality Reduction:
  
  -PCA (Principal Component Analysis)
  
  -Precomputed t-SNE visualization
  
  -Patient Clustering:
  Compare multiple clustering algorithms:
  
  -K-Means
  
  -DBSCAN
  
  -Hierarchical Clustering
  
  -Includes t-SNE cluster overlays and silhouette-based comparisons.
  
  -Association Rule Mining:
  Apriori-based disease association discovery with lift, confidence, and support metrics.

Dashboard Workflow

  -Raw Data → Cleaning:
  
  -Remove irrelevant columns
  
  -Standardize numerical features
  
  -Clip extreme values
  
  -Encode categorical variables
  
  -Analysis Layer:
  
  -Statistics
  
  -Feature correlations
  
  -Principal components
  
  -Clustering
  
  -Association patterns
  
  -Visualization Layer:
  
  -Interactive charts (Plotly), tables, metrics, and gradient-styled UI.

Page-by-Page Overview

  1️⃣ Overview
  
  Project introduction
  
  Data preview (first 100 rows)
  
  Key dataset statistics
  
  2️⃣ Summary Statistics
  
  Descriptive stats (scaled & unscaled)
  
  Histograms & pie charts
  
  Outlier treatment notes
  
  3️⃣ Feature Selection
  
  Correlation heatmap
  
  Adjustable correlation threshold slider
  
  List of selected predictive features
  
  CSV download option
  
  4️⃣ Dimensionality Reduction
  
  PCA visualization of selected features
  
  Precomputed t-SNE projection (threshold-specific)
  
  5️⃣ Patient Clustering
  
  K-Means, DBSCAN, and Hierarchical analysis
  
  Parameter controls
  
  Silhouette comparison
  
  t-SNE cluster overlays
  
  Cluster profile table
  
  6️⃣ Association Rules
  
  Disease pattern discovery using Apriori
  
  Top 10 strongest rules
  
  Support / confidence / lift summary
  
  Insight explanations

Deployment

  -The dashboard is publicly hosted on HuggingFace Spaces using the Streamlit template:
  
  -Live App:
  https://huggingface.co/spaces/Megha-Harsha/Hospital-Patient-Comorbidity-and-Readmission-Dashboard
  
  -Note:
  The app loads and processes a large dataset during startup.
  Please wait 10-12 minutess for the dashboard to fully initialize.

Tech Stack:

  -Python
  
  -Streamlit
  
  -Pandas / NumPy
  
  -Plotly
  
  -Scikit-Learn
  
  -Apriori (mlxtend)
  
  -HuggingFace Spaces (Deployment)
