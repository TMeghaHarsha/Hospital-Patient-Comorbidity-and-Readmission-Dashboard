import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_id_mappings():
    """
    Load ID mappings from the mapping file.
    
    Returns:
        dict: Dictionary containing mappings for different ID types
    """
    mappings = {}
    
    # Read the mapping file
    with open('data/IDS_mapping.csv', 'r') as f:
        content = f.read()
    
    # Split by sections
    sections = content.split('\n\n')
    
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            if len(lines) > 1:
                section_name = lines[0].split(',')[0]
                mapping_dict = {}
                
                for line in lines[1:]:
                    if ',' in line and line.strip():
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                mapping_dict[key] = value
                
                if mapping_dict:
                    mappings[section_name] = mapping_dict
    
    return mappings

def load_and_clean_data():
    """
    Load and clean the diabetic data for analysis.
    
    Returns:
        tuple: (cleaned_dataframe, numerical_features_list, id_mappings)
    """
    # Load the data
    df = pd.read_csv('data/diabetic_data.csv')
    
    # Load ID mappings
    id_mappings = load_id_mappings()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop columns with high percentage of missing values
    columns_to_drop = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=columns_to_drop)
    
    # Drop columns not useful for analysis
    columns_to_drop = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=columns_to_drop)
    
    # Select numerical features for analysis (define before using)
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    
    # Note: ID mappings are no longer applied - we use ID columns directly
    
    # Ensure numerical features exist and are numeric
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Intelligent imputation to preserve all data
    # Categorical imputation
    if 'race' in df.columns:
        race_mode = df['race'].mode()[0] if not df['race'].mode().empty else 'Unknown'
        df['race'] = df['race'].fillna(race_mode)
    
    # Impute diagnosis columns with 'Unknown'
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[diag_col] = df[diag_col].fillna('Unknown')
    
    # Numerical imputation using median
    for feature in numerical_features:
        if feature in df.columns:
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value)
    
    # Clip outliers for specific features
    if 'number_emergency' in df.columns:
        df['number_emergency'] = df['number_emergency'].clip(upper=15)
    if 'number_outpatient' in df.columns:
        df['number_outpatient'] = df['number_outpatient'].clip(upper=15)
    if 'num_medications' in df.columns:
        df['num_medications'] = df['num_medications'].clip(upper=50)
    
    # Scale numerical features after clipping
    df_scaled = df.copy()
    scaler = StandardScaler()
    # Scale all numerical features together
    available_numerical_features = [f for f in numerical_features if f in df_scaled.columns]
    if available_numerical_features:
        scaled_values = scaler.fit_transform(df_scaled[available_numerical_features])
        # Convert back to DataFrame to preserve column names and index
        df_scaled[available_numerical_features] = pd.DataFrame(
            scaled_values, 
            columns=available_numerical_features, 
            index=df_scaled.index
        )
    
    # Encode the target variable 'readmitted'
    df['readmitted_bool'] = df['readmitted'].apply(
        lambda x: 0 if x == 'NO' else 1
    )
    df_scaled['readmitted_bool'] = df_scaled['readmitted'].apply(
        lambda x: 0 if x == 'NO' else 1
    )
    
    return df, df_scaled, numerical_features, id_mappings
