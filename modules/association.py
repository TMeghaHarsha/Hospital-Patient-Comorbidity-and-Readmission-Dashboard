import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def get_disease_name(diag_code):
    """
    Convert ICD-9 diagnosis codes to disease names.
    
    Args:
        diag_code: ICD-9 diagnosis code
        
    Returns:
        str: Disease name or original code if not found
    """
    # Common ICD-9 to disease mappings
    disease_mappings = {
        '250': 'Diabetes mellitus',
        '250.0': 'Diabetes mellitus without complications',
        '250.1': 'Diabetes with ketoacidosis',
        '250.2': 'Diabetes with hyperosmolarity',
        '250.3': 'Diabetes with other coma',
        '250.4': 'Diabetes with renal complications',
        '250.5': 'Diabetes with ophthalmic complications',
        '250.6': 'Diabetes with neurological complications',
        '250.7': 'Diabetes with peripheral circulatory complications',
        '250.8': 'Diabetes with other specified complications',
        '250.9': 'Diabetes with unspecified complications',
        '250.83': 'Diabetes with other specified complications',
        '250.01': 'Diabetes mellitus without complications',
        '250.43': 'Diabetes with other specified complications',
        '276': 'Disorders of fluid, electrolyte, and acid-base balance',
        '255': 'Disorders of adrenal glands',
        '648': 'Other current conditions in the mother classifiable elsewhere',
        '250': 'Diabetes mellitus',
        '403': 'Hypertensive chronic kidney disease',
        '996': 'Complications of transplanted organ',
        'V27': 'Outcome of delivery',
        '8': 'Other specified diseases and conditions',
        '276': 'Disorders of fluid, electrolyte, and acid-base balance',
        '250.01': 'Diabetes mellitus without complications',
        '250.43': 'Diabetes with other specified complications',
        '403': 'Hypertensive chronic kidney disease',
        '996': 'Complications of transplanted organ',
        'V27': 'Outcome of delivery',
        '8': 'Other specified diseases and conditions'
    }
    
    # Try exact match first
    if diag_code in disease_mappings:
        return disease_mappings[diag_code]
    
    # Try partial match for main categories
    main_category = diag_code.split('.')[0] if '.' in diag_code else diag_code
    if main_category in disease_mappings:
        return disease_mappings[main_category]
    
    # Return original code if no mapping found
    return f"ICD-9: {diag_code}"

def find_association_rules(df):
    """
    Find association rules using the Apriori algorithm on diagnosis data.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame: Association rules sorted by lift
    """
    # Focus on diagnosis columns
    diagnosis_columns = ['diag_1', 'diag_2', 'diag_3']
    
    # Create transactions list - each patient's diagnoses
    transactions = []
    for _, row in df[diagnosis_columns].iterrows():
        # Get non-null diagnoses for this patient
        patient_diagnoses = [str(diag) for diag in row.values if pd.notna(diag) and str(diag) != 'nan']
        if patient_diagnoses:  # Only add if there are diagnoses
            transactions.append(patient_diagnoses)
    
    # Use TransactionEncoder to one-hot encode
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Run apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        # Convert frozensets to disease names
        def convert_to_disease_names(frozenset_items):
            disease_names = []
            for item in frozenset_items:
                disease_name = get_disease_name(item)
                disease_names.append(disease_name)
            return ', '.join(disease_names)
        
        rules["antecedents"] = rules["antecedents"].apply(convert_to_disease_names)
        rules["consequents"] = rules["consequents"].apply(convert_to_disease_names)
        
        # Sort by lift in descending order
        rules = rules.sort_values('lift', ascending=False)
        return rules
    else:
        # Return empty DataFrame with expected columns if no rules found
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])