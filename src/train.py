import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
from tqdm import tqdm
from src.feature_selection import preprocess_and_select_features
from src.losses import FocalLoss
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LEARNING_RATE, BATCH_SIZE, EPOCHS

def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Train the model

    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs.view(-1), torch.FloatTensor(y_val).view(-1))
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss.item():.4f}')
        
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping training")
            break

    return model

def train_final_model(best_model_class, best_params, X_train_processed, y_train_resampled):
    """
    Train the final model using best paramsq

    Args:
        best_model_class: Class of the best performing model
        best_params: Best hyperparameters for the model
        X_train_processed: Processed training features
        y_train_resampled: Resampled training labels

    Returns:
        Trained final model
    """
    final_model = best_model_class(**best_params)
    final_model = train_model(final_model, torch.FloatTensor(X_train_processed), torch.FloatTensor(y_train_resampled),
                            torch.FloatTensor(X_train_processed), torch.FloatTensor(y_train_resampled), epochs=100)
    return final_model

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