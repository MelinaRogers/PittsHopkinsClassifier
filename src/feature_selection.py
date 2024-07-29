import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, fbeta_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc

def select_features(X, y, k=100):
    """
    Pick the top k features using mutual information.

    Args:
    X: Our feature matrix
    y: Our target variable
    k: How many features 

    Returns:
    The selected features and their indices
    """
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_feature_indices = selector.get_support(indices=True)
    return X_selected, selected_feature_indices

def preprocess_and_select_features(X, y, n_features=100):
    """
    Clean up data and choosebe st features

    Args:
    X: Our feature matrix
    y: Our target variable
    n_features: How many features 

    Returns:
    Processed and resampled data, and tools used to process
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    return X_resampled, y_resampled, selector, scaler