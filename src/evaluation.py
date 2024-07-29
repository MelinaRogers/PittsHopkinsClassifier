import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, fbeta_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt 
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc

def make_predictions(final_model, X_test_processed):
    """
    Predicts based on final model 

    Args:
    final_model (torch.nn.Module): The trained NN model
    X_test_processed (np.array): Processed test features

    Returns:
    tuple: (test_probs, test_preds) where test_probs are the predicted probabilities
           and test_preds are the binary predictions
    """
    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(torch.FloatTensor(X_test_processed))
        test_probs = torch.sigmoid(test_outputs).numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
    return test_probs, test_preds

def generate_plots(y_test, test_probs, final_model, X_test_processed):
    """
    Generate and save various evaluation plots for report

    Args:
    y_test (np.array): True labels for the test set
    test_probs (np.array): Predicted probabilities for the test set
    final_model (torch.nn.Module): The trained neural network model
    X_test_processed (np.array): Processed test features
    """
    # Calibration plot
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration plot')
    plt.savefig('calibration_plot_c.png')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_c.png')
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    average_precision = average_precision_score(y_test, test_probs)
    plt.figure(figsize=(10, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.savefig('precision_recall_curve_c.png')
    plt.close()

def confidence_score(probabilities, feature_names, key_hpo_terms):
    """
    Classification threshold and prediction 
    
    Args:
    probabilities (np.array): Model prediction probabilities
    feature_names (list): List of feature names
    key_hpo_terms (list): List of key HPO terms for the disease
    
    Returns:
    np.array: Confidence scores
    """
    hpo_scores = []
    for features in feature_names:
        hpo_score = sum(1 for term in key_hpo_terms if term in features) / len(key_hpo_terms)
        hpo_scores.append(hpo_score)
    hpo_scores = np.array(hpo_scores)
    
    if len(hpo_scores) != len(probabilities):
        hpo_scores = np.full(len(probabilities), np.mean(hpo_scores))
    
    return (probabilities + hpo_scores) / 2

def tune_threshold_and_predict(y_test, isotonic_proba, confidences):
    """
    Classification threshold tuning and predictions based on it 

    Args:
    y_test (np.array): True labels for the test set
    isotonic_proba (np.array): Calibrated probabilities from isotonic regression
    confidences (np.array): Confidence scores

    Returns:
    tuple: (optimal_threshold, final_predictions, final_predictions_top_100)
    """
    optimal_threshold = tune_threshold(y_test, isotonic_proba, n_tests=100, min_recall=0.80)
    final_predictions = (isotonic_proba >= optimal_threshold).astype(int)

    top_100_indices = np.argsort(confidences)[-100:]
    final_predictions_top_100 = np.zeros_like(final_predictions)
    final_predictions_top_100[top_100_indices] = 1

    return optimal_threshold, final_predictions, final_predictions_top_100

def feature_importance(model, X):
    """
    Calculate feature importance for given model 

    Args:
    model (torch.nn.Module): The trained neural network model
    X (np.array): Input features

    Returns:
    list: Feature importances
    """
    X = torch.FloatTensor(X)
    importances = []
    model.eval()
    with torch.no_grad():
        baseline = model(X).mean()
        for i in range(X.shape[1]):
            X_copy = X.clone()
            X_copy[:, i] = 0
            output = model(X_copy).mean()
            importance = abs(baseline - output)
            importances.append(importance.item())
    return importances

def tune_threshold(y_true, y_pred_proba, n_tests=100, min_recall=0.80):
    """
    Tune the classification threshold
    
    Args:
    y_true (np.array): True labels
    y_pred_proba (np.array): Predicted probabilities
    n_tests (int): Number of tests available
    min_recall (float): Minimum required recall
    
    Returns:
    float: Optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    for i, (precision, recall, threshold) in enumerate(zip(precisions, recalls, thresholds)):
        predicted_positives = (y_pred_proba >= threshold).sum()
        if recall >= min_recall and predicted_positives <= n_tests:
            optimal_threshold = threshold
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label='Precision-Recall curve')
    plt.axvline(recall, color='r', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.legend()
    plt.savefig('precision_recall_curve_.png')
    plt.close()
    
    return optimal_threshold

def calibrate_model(model, X_train, y_train, X_test):
    """
    Calibrate the model using Platt Scaling and Isotonic Regression to 
    see what works best 
    
    Args:
    model: Model to calibrate
    X_train (np.array): Training features
    y_train (np.array): Training labels
    X_test (np.array): Test features
    
    Returns:
    tuple: (platt_calibrated, isotonic_calibrated, platt_proba, isotonic_proba)
    """
    platt_calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
    platt_calibrated.fit(X_train, y_train)
    platt_proba = platt_calibrated.predict_proba(X_test)[:, 1]

    isotonic_calibrated = CalibratedClassifierCV(estimator=model, method='isotonic', cv=5)
    isotonic_calibrated.fit(X_train, y_train)
    isotonic_proba = isotonic_calibrated.predict_proba(X_test)[:, 1]

    return platt_calibrated, isotonic_calibrated, platt_proba, isotonic_proba

def plot_calibration_curves(y_test, y_pred_proba, platt_proba, isotonic_proba):
    """
    Plot calibration curves for the original and calibrated models
    
    Args:
    y_test (np.array): True labels for the test set
    y_pred_proba (np.array): Original model probabilities
    platt_proba (np.array): Platt scaling probabilities
    isotonic_proba (np.array): Isotonic regression probabilities
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for proba, label in [(y_pred_proba, 'Original'),
                         (platt_proba, 'Platt Scaling'),
                         (isotonic_proba, 'Isotonic Regression')]:
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        ax1.plot(prob_pred, prob_true, "s-", label=label)
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.hist(y_pred_proba, range=(0, 1), bins=10, label="Original", histtype="step", lw=2)
    ax2.hist(platt_proba, range=(0, 1), bins=10, label="Platt Scaling", histtype="step", lw=2)
    ax2.hist(isotonic_proba, range=(0, 1), bins=10, label="Isotonic Regression", histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig('calibration_curves.png')
    plt.close()

def evaluate_and_report(y_test, final_predictions_top_100, optimal_threshold):
    """
    Evaluate the model and report various metrics

    Args:
    y_test (np.array): True labels for the test set
    final_predictions_top_100 (np.array): Final predictions for top 100 cases
    optimal_threshold (float): Optimal classification threshold
    """
    cm = confusion_matrix(y_test, final_predictions_top_100)
    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(y_test, final_predictions_top_100)
    recall = recall_score(y_test, final_predictions_top_100)
    f2_score = fbeta_score(y_test, final_predictions_top_100, beta=2)
    predicted_positives = final_predictions_top_100.sum()

    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2 Score: {f2_score:.4f}")
    print(f"Number of predicted positive cases: {predicted_positives}")
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions_top_100))

def analyze_feature_importance(final_model, X_test_processed, all_feature_names, selected_indices):
    """
    Analyze and visualize feature importance

    Args:
    final_model (torch.nn.Module): The trained neural network model
    X_test_processed (np.array): Processed test features
    all_feature_names (list): List of all feature names
    selected_indices (list): Indices of selected features
    """
    importances = feature_importance(final_model, X_test_processed)
    
    importances = np.array(importances)
    
    sorted_idx = np.argsort(importances)
    sorted_importances = importances[sorted_idx]
    sorted_feature_names = [all_feature_names[selected_indices[i]] for i in sorted_idx]

    print("Top 20 important features:")
    for name, importance in zip(sorted_feature_names[-20:], sorted_importances[-20:]):
        print(f"{name}: {importance:.4f}")

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(sorted_importances))
    plt.barh(y_pos, sorted_importances)
    plt.yticks(y_pos, sorted_feature_names)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances_f.png')
    plt.close()