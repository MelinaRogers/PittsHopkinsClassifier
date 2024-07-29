import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
from .feature_selection import preprocess_and_select_features, select_features
import torch 

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_data():
    """
    Loads the feature and target data 
    
    Returns:
        tuple: A tuple containing:
            - X (np.array): Feature matrix
            - y (np.array): Target vector
    """
    X = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'X_feature_f.npy'))
    y = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'y_target_f.npy'))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, y

def load_feature_names():
    """
    Loads the feature names and HPO terms
    
    Returns:
        tuple: A tuple containing:
            - all_feature_names (list): List of all feature names
            - key_hpo_terms (list): List of HPO terms for Pitt Hopkins Syndrome
    """
    with open(os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_names_f.json'), 'r') as f:
        feature_names = json.load(f)
    
    all_feature_names = (feature_names['weighted_hpo_features'] + 
                     feature_names['text_features'] + 
                     feature_names['embedding_features'])

    # Define key HPO terms for Pitt Hopkins Syndrome
    key_hpo_terms = [
        "HP:0000750",  # Delayed speech and language development
        "HP:0002079",  # Breathing abnormality
        "HP:0000322",  # Short stature
        "HP:0004322",  # Short philtrum
        "HP:0010509",  # Severe intellectual disability
        "HP:0008082"   # Abnormal gait
    ]        
    return all_feature_names, key_hpo_terms

def preprocess_test_data(X_test, scaler, selector):
    """
    Preprocessing the test data
    
    Args:
        X_test (np.array): Test feature matrix
        scaler (StandardScaler): Fitted StandardScaler object
        selector: Fitted feature selector object
    
    Returns:
        np.array: Preprocessed test data
    """
    X_test_scaled = scaler.transform(X_test)
    X_test_processed = selector.transform(X_test_scaled)
    return X_test_processed

def save_model_and_features(final_model, all_feature_names, selected_indices):
    """
    Save trained model and selected feature names 
    
    Args:
        final_model: Trained model object
        all_feature_names (list): List of all feature names
        selected_indices (list): Indices of selected features
    """
    torch.save(final_model.state_dict(), 'final_model.pth')
    selected_feature_names = [all_feature_names[i] for i in selected_indices]
    with open('selected_feature_names_c.json', 'w') as f:
        json.dump(selected_feature_names, f)

def process_data():
    """
    main function for all data processing, including loading,
    feature selection, and preprocessing
    
    Returns:
        tuple: A tuple containing processed data and metadata:
            - X_train (np.array): Raw training features
            - X_test (np.array): Raw test features
            - y_train (np.array): Raw training targets
            - y_test (np.array): Raw test targets
            - all_feature_names (list): List of all feature names
            - key_hpo_terms (list): List of key HPO terms
            - selector: Fitted feature selector object
            - scaler (StandardScaler): Fitted StandardScaler object
            - X_train_processed (np.array): Processed training features
            - y_train_resampled (np.array): Resampled training targets
            - X_test_processed (np.array): Processed test features
            - selected_indices (list): Indices of selected features
    """
    X, y = load_data()
    all_feature_names, key_hpo_terms = load_feature_names()
    X_selected, selected_indices = select_features(X, y, k=100)
    print(f"Number of selected features: {X_selected.shape[1]}")
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing all data
    X_train_processed, y_train_resampled, selector, scaler = preprocess_and_select_features(X_train, y_train, n_features=100)
    X_test_processed = preprocess_test_data(X_test, scaler, selector)
    return X_train, X_test, y_train, y_test, all_feature_names, key_hpo_terms, selector, scaler, X_train_processed, y_train_resampled, X_test_processed, selected_indices