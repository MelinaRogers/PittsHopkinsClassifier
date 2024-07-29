from sklearn.model_selection import StratifiedKFold
import torch
from src.train import train_model
import numpy as np 
from sklearn.metrics import fbeta_score
from src.feature_selection import preprocess_and_select_features

def cross_validate_with_preprocessing(X, y, model_class, n_splits=5, n_features=100, **model_params):
    """
    Perform cross-validation with preprocessing 

    Args:
        X: Feature matrix
        y: Target variable
        model_class: Model class to use
        n_splits: Number of cross-validation splits
        n_features: Number of features to select
        **model_params: Additional model parameters

    Returns:
        Mean and standard deviation of F2 scores
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = y[train_index], y[val_index]

        X_fold_train_processed, y_fold_train_resampled, selector, scaler = preprocess_and_select_features(X_fold_train, y_fold_train, n_features)
        
        X_fold_val_scaled = scaler.transform(X_fold_val)
        X_fold_val_processed = selector.transform(X_fold_val_scaled)

        model = model_class(**model_params)
        model = train_model(model, torch.FloatTensor(X_fold_train_processed), torch.FloatTensor(y_fold_train_resampled),
                            torch.FloatTensor(X_fold_val_processed), torch.FloatTensor(y_fold_val), epochs=50)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_fold_val_processed))
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        
        score = fbeta_score(y_fold_val, val_preds.view(-1).numpy(), beta=2)
        scores.append(score)

    return np.mean(scores), np.std(scores)