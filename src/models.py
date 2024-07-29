import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.train import cross_validate_with_preprocessing

class AttentionLayer(nn.Module):
    """Attention layer for neural network"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * attention_weights, dim=1)

class AttentionNN(nn.Module):
    """Neural network with attention mechanism"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, dropout_rate=0.5):
        super(AttentionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = AttentionLayer(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return x
    
class RegularizedNN(nn.Module):
    """Regularized neural network"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, dropout_rate=0.5):
        super(RegularizedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for neural network models for API"""
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        probs = torch.sigmoid(outputs).numpy()
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
class StackingEnsemble:
    """Stacking ensemble model"""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
    
    def fit(self, X, y):
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
        
        meta_features = self._generate_meta_features(X)
        
        self.meta_model.fit(meta_features, y)
    
    def predict(self, X):
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict_proba(meta_features)[:, 1]
    
    def _generate_meta_features(self, X):
        meta_features = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                meta_features.append(model.predict_proba(X)[:, 1])
            elif isinstance(model, NeuralNetworkWrapper):
                meta_features.append(model.predict_proba(X)[:, 1])
        return np.column_stack(meta_features)

def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters"""
    input_dim = X_train.shape[1]
    hidden_dims = [64, 128, 256]
    dropout_rates = [0.3, 0.5, 0.7]

    best_score = -np.inf
    best_params = None
    best_model_class = None

    for model_class in [RegularizedNN, AttentionNN]:
        print(f"Tuning {model_class.__name__}")
        for hidden_dim in hidden_dims:
            for dropout_rate in dropout_rates:
                model_params = {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'num_classes': 1,
                    'dropout_rate': dropout_rate
                }
                
                mean_score, std_score = cross_validate_with_preprocessing(
                    X_train, y_train, model_class, 
                    n_splits=5, n_features=100, 
                    **model_params
                )
                print(f"Hidden dim: {hidden_dim}, Dropout rate: {dropout_rate}, Mean F2: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = model_params.copy()
                    best_model_class = model_class

    print(f"Best model: {best_model_class.__name__}")
    print(f"Best parameters: {best_params}")
    return best_model_class, best_params

def create_stacking_ensemble(X_train_processed, y_train_resampled, final_model):
    """Create stacking ensemble model"""
    nn_wrapper = NeuralNetworkWrapper(final_model)
    nn_wrapper.fit(X_train_processed, y_train_resampled)

    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        XGBClassifier(n_estimators=100, random_state=42),
        nn_wrapper
    ]
    meta_model = LogisticRegression()
    stacking_ensemble = StackingEnsemble(base_models, meta_model)
    stacking_ensemble.fit(X_train_processed, y_train_resampled)
    return stacking_ensemble