from src.data_utils import process_data, save_model_and_features
from src.models import tune_hyperparameters, create_stacking_ensemble
from src.train import train_final_model
from src.evaluation import calibrate_model, plot_calibration_curves, make_predictions, confidence_scores, tune_threshold_and_predict, evaluate_and_report, analyze_feature_importance, generate_plots
from src.models import NeuralNetworkWrapper

def run_pipeline():
    """
    Full ml pipeline with data processing, model training, 
    evaluation, and analysis

    Returns:
    tuple: Contains the final trained model, processed test data, test labels, 
           predicted probabilities, and final top 100 predictions
    """
    X_train, X_test, y_train, y_test, all_feature_names, key_hpo_terms, _, _, X_train_processed, y_train_resampled, X_test_processed, selected_indices = process_data()
    print("Data processing completed")

    best_model_class, best_params = tune_hyperparameters(X_train, y_train)
    print("Hyperparameter tuning completed")

    final_model = train_final_model(best_model_class, best_params, X_train_processed, y_train_resampled)
    print("Final model trained")

    stacking_ensemble = create_stacking_ensemble(X_train_processed, y_train_resampled, final_model)
    print("Stacking ensemble created and trained")

    test_probs, test_preds = make_predictions(final_model, X_test_processed)
    stacking_predictions = stacking_ensemble.predict(X_test_processed)
    print("Predictions made for both final model and stacking ensemble")

    nn_wrapper = NeuralNetworkWrapper(final_model)
    platt_calibrated, isotonic_calibrated, platt_proba, isotonic_proba = calibrate_model(
        nn_wrapper, X_train_processed, y_train_resampled, X_test_processed
    )
    print("Model calibrated")

    plot_calibration_curves(y_test, test_probs, platt_proba, isotonic_proba)
    print("Calibration curves plotted")

    confidences = confidence_scores(isotonic_proba, all_feature_names, key_hpo_terms)
    print("Confidence scores calculated")

    optimal_threshold, final_predictions, final_predictions_top_100 = tune_threshold_and_predict(
        y_test, isotonic_proba, confidences
    )
    print("Threshold tuned and final predictions made")

    evaluate_and_report(y_test, final_predictions_top_100, optimal_threshold)
    print("Evaluation completed and results reported")

    analyze_feature_importance(final_model, X_test_processed, all_feature_names, selected_indices)
    print("Feature importance analyzed")

    generate_plots(y_test, test_probs, final_model, X_test_processed)
    print("Plots generated")

    save_model_and_features(final_model, all_feature_names, selected_indices)
    print("Model and selected feature names saved :) Yay")

    return final_model, X_test_processed, y_test, test_probs, final_predictions_top_100