import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

from config.paths import DATA_DIR
from preprocessing.clinical_data.clinical_data_preprocessing import preprocess_data


def remove_low_variance_features(train_df, feature_cols, variance_threshold=0.01):
    """
    Remove features with variance below the specified threshold.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    feature_cols : list
        List of feature column names
    variance_threshold : float
        Minimum variance threshold (default: 0.01)

    Returns
    -------
    tuple
        (filtered_feature_cols, variance_stats)
    """
    # Calculate variance for each feature in training data
    feature_variances = train_df[feature_cols].var()

    # Identify low variance features
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    high_variance_features = feature_variances[feature_variances >= variance_threshold].index.tolist()

    print(f"\nVariance Filtering Results:")
    print(f"Features with variance < {variance_threshold}: {len(low_variance_features)}")
    print(f"Features with variance >= {variance_threshold}: {len(high_variance_features)}")
    print(f"Percentage of features removed: {len(low_variance_features) / len(feature_cols) * 100:.1f}%")

    if len(low_variance_features) > 0:
        print(f"\nSample of removed low-variance features:")
        for i, feature in enumerate(low_variance_features[:5]):  # Show first 5
            print(f"  {feature}: variance = {feature_variances[feature]:.6f}")
        if len(low_variance_features) > 5:
            print(f"  ... and {len(low_variance_features) - 5} more")

    # Create variance statistics summary
    variance_stats = {
        'total_features': len(feature_cols),
        'removed_features': len(low_variance_features),
        'kept_features': len(high_variance_features),
        'min_variance': feature_variances.min(),
        'max_variance': feature_variances.max(),
        'mean_variance': feature_variances.mean(),
        'removed_feature_names': low_variance_features
    }

    return high_variance_features, variance_stats


def tune_hyperparameters(train_cox, val_cox, time_col, event_col, penalizer_values):
    """
    Tune hyperparameters using validation set.

    Parameters
    ----------
    train_cox : pd.DataFrame
        Training data with features, time, and event columns
    val_cox : pd.DataFrame
        Validation data with features, time, and event columns
    time_col : str
        Name of survival time column
    event_col : str
        Name of event column
    penalizer_values : list
        List of penalizer values to try

    Returns
    -------
    tuple
        (best_penalizer, best_val_c_index, results_dict)
    """
    results = {}
    best_c_index = -np.inf
    best_penalizer = None

    print("\nHyperparameter Tuning Results:")
    print("-" * 50)

    for penalizer in penalizer_values:
        # Fit model with current penalizer
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(train_cox, duration_col=time_col, event_col=event_col)

        # Evaluate on validation set
        val_c_index = cph.score(val_cox, scoring_method='concordance_index')

        results[penalizer] = val_c_index
        print(f"Penalizer: {penalizer:.4f} | Validation C-index: {val_c_index:.4f}")

        # Update best parameters
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            best_penalizer = penalizer

    print("-" * 50)
    print(f"Best penalizer: {best_penalizer:.4f} with validation C-index: {best_c_index:.4f}")

    return best_penalizer, best_c_index, results


def run_cox_regression(train_path: str, val_path: str, test_path: str, time_col: str = 'Survival.time',
                       event_col: str = 'deadstatus.event', variance_threshold: float = 0.01):
    """
    Run Cox Proportional Hazards regression with hyperparameter tuning and evaluate C-index.

    Parameters
    ----------
    train_path : str
        Path to preprocessed training data
    val_path : str
        Path to preprocessed validation data
    test_path : str
        Path to preprocessed test data
    time_col : str
        Name of survival time column
    event_col : str
        Name of event/censoring column (1 = event, 0 = censored)
    variance_threshold : float
        Minimum variance threshold for feature selection (default: 0.01)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Remove non-feature columns
    id_cols = ['PatientID'] if 'PatientID' in train_df.columns else []
    feature_cols = [col for col in train_df.columns
                    if col not in [time_col, event_col] + id_cols]

    print(f"\nInitial number of features: {len(feature_cols)}")

    # Remove low variance features
    filtered_feature_cols, variance_stats = remove_low_variance_features(
        train_df, feature_cols, variance_threshold
    )

    print(f"Final number of features after variance filtering: {len(filtered_feature_cols)}")

    # Prepare data for Cox regression (needs time and event columns)
    train_cox = train_df[filtered_feature_cols + [time_col, event_col]].copy()
    val_cox = val_df[filtered_feature_cols + [time_col, event_col]].copy()
    test_cox = test_df[filtered_feature_cols + [time_col, event_col]].copy()

    print(f"Train samples: {len(train_cox)}")
    print(f"Val samples: {len(val_cox)}")
    print(f"Test samples: {len(test_cox)}")

    # Define hyperparameter search space
    penalizer_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    # Tune hyperparameters using validation set
    best_penalizer, best_val_c_index, tuning_results = tune_hyperparameters(
        train_cox, val_cox, time_col, event_col, penalizer_values
    )

    # Train final model with best hyperparameters
    print(f"\nTraining final model with best penalizer: {best_penalizer}")
    cph = CoxPHFitter(penalizer=best_penalizer)
    cph.fit(train_cox, duration_col=time_col, event_col=event_col)

    # Print model summary
    print("\nCox Model Summary (showing first 10 features):")
    print(cph.summary.iloc[:10, :3])  # Show first 10 rows and 3 columns of summary

    # Calculate C-index on all sets
    train_c_index = cph.score(train_cox, scoring_method='concordance_index')
    val_c_index = cph.score(val_cox, scoring_method='concordance_index')
    test_c_index = cph.score(test_cox, scoring_method='concordance_index')

    print(f"\nFinal C-index Scores:")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index: {val_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")

    return cph, train_c_index, val_c_index, test_c_index, best_penalizer


if __name__ == '__main__':
    train_ci_scores = []
    val_ci_scores = []
    test_ci_scores = []
    best_penalizers = []

    for seed in range(50):
        print(f"\n{'=' * 60}")
        print(f"Running iteration {seed + 1}/50 with seed {seed}")
        print(f"{'=' * 60}")

        preprocess_data(seed=seed)  # Different train/val/test splits

        # Run Cox regression with hyperparameter tuning
        cph_model, train_ci, val_ci, test_ci, best_penalizer = run_cox_regression(
            train_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_train.csv'),
            val_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_val.csv'),
            test_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_test.csv'),
            time_col='Survival.time',
            event_col='deadstatus.event',
            variance_threshold=0.01  # Remove features with variance < 0.01
        )

        train_ci_scores.append(train_ci)
        val_ci_scores.append(val_ci)
        test_ci_scores.append(test_ci)
        best_penalizers.append(best_penalizer)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nMost frequently selected penalizer values:")
    unique_penalizers, counts = np.unique(best_penalizers, return_counts=True)
    for pen, count in sorted(zip(unique_penalizers, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Penalizer {pen}: selected {count} times ({count / 50 * 100:.1f}%)")

    print(f"\nTrain CI scores: Mean = {np.mean(train_ci_scores):.4f} ± {np.std(train_ci_scores):.4f}")
    print(f"Val CI scores: Mean = {np.mean(val_ci_scores):.4f} ± {np.std(val_ci_scores):.4f}")
    print(f"Test CI scores: Mean = {np.mean(test_ci_scores):.4f} ± {np.std(test_ci_scores):.4f}")