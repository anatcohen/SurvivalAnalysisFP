import os
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import ParameterGrid

from config.paths import DATA_DIR
from preprocessing.clinical_data.clinical_data_preprocessing import preprocess_data


def remove_low_variance_features(train_df, test_df, feature_cols, variance_threshold=0.01):
    """
    Remove features with variance below the specified threshold.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
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


def prepare_survival_data(df, feature_cols, time_col, event_col):
    """
    Prepare data for scikit-survival format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and survival information
    feature_cols : list
        List of feature column names
    time_col : str
        Name of survival time column
    event_col : str
        Name of event column

    Returns
    -------
    tuple
        (X, y) where X is feature matrix and y is structured array for survival
    """
    X = df[feature_cols].values

    # Create structured array for survival data
    y = Surv.from_dataframe(event_col, time_col, df)

    return X, y


def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, random_state=42):
    """
    Tune hyperparameters using validation set.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : structured array
        Training survival data
    X_val : np.ndarray
        Validation features
    y_val : structured array
        Validation survival data
    param_grid : dict
        Dictionary of parameters to search
    random_state : int
        Random state for reproducibility

    Returns
    -------
    tuple
        (best_params, best_val_c_index, results_list)
    """
    results = []
    best_c_index = -np.inf
    best_params = None

    print("\nHyperparameter Tuning Results:")
    print("-" * 70)
    print(f"{'n_estimators':<12} {'max_depth':<10} {'min_samples_split':<18} {'Val C-index':<12}")
    print("-" * 70)

    # Create parameter combinations
    param_combinations = list(ParameterGrid(param_grid))

    for params in param_combinations:
        # Create and train model with current parameters
        rsf = RandomSurvivalForest(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            n_jobs=-1,
            random_state=random_state
        )

        rsf.fit(X_train, y_train)

        # Evaluate on validation set
        val_c_index = rsf.score(X_val, y_val)

        results.append({
            'params': params,
            'val_c_index': val_c_index
        })

        # Handle None value for max_depth in printing
        max_depth_str = str(params['max_depth']) if params['max_depth'] is not None else 'None'
        print(f"{params['n_estimators']:<12} {max_depth_str:<10} "
              f"{params['min_samples_split']:<18} {val_c_index:<12.4f}")

        # Update best parameters
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            best_params = params

    print("-" * 70)
    print(f"Best parameters: {best_params}")
    print(f"Best validation C-index: {best_c_index:.4f}")

    return best_params, best_c_index, results


def run_random_survival_forest(train_path: str, val_path: str, test_path: str,
                               time_col: str = 'Survival.time',
                               event_col: str = 'deadstatus.event',
                               variance_threshold: float = 0.01,
                               random_state: int = 42):
    """
    Run Random Survival Forest with hyperparameter tuning and evaluate C-index.

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
    random_state : int
        Random state for reproducibility
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
        train_df, test_df, feature_cols, variance_threshold
    )

    print(f"Final number of features after variance filtering: {len(filtered_feature_cols)}")

    # Prepare data for Random Survival Forest
    X_train, y_train = prepare_survival_data(train_df, filtered_feature_cols, time_col, event_col)
    X_val, y_val = prepare_survival_data(val_df, filtered_feature_cols, time_col, event_col)
    X_test, y_test = prepare_survival_data(test_df, filtered_feature_cols, time_col, event_col)

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2', 0.5]
    }

    # Tune hyperparameters using validation set
    best_params, best_val_c_index, tuning_results = tune_hyperparameters(
        X_train, y_train, X_val, y_val, param_grid, random_state
    )

    # Train final model with best hyperparameters
    print(f"\nTraining final model with best parameters: {best_params}")
    rsf = RandomSurvivalForest(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params.get('min_samples_leaf', 1),
        max_features=best_params.get('max_features', 'sqrt'),
        n_jobs=-1,
        random_state=random_state
    )

    rsf.fit(X_train, y_train)

    # Calculate C-index on all sets
    train_c_index = rsf.score(X_train, y_train)
    val_c_index = rsf.score(X_val, y_val)
    test_c_index = rsf.score(X_test, y_test)

    print(f"\nFinal C-index Scores:")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index: {val_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")

    return rsf, train_c_index, val_c_index, test_c_index, best_params


if __name__ == '__main__':
    train_ci_scores = []
    val_ci_scores = []
    test_ci_scores = []
    best_params_list = []

    for seed in range(50):
        print(f"\n{'=' * 60}")
        print(f"Running iteration {seed + 1}/50 with seed {seed}")
        print(f"{'=' * 60}")

        preprocess_data(seed=seed)  # Different train/val/test splits

        # Run Random Survival Forest with hyperparameter tuning
        rsf_model, train_ci, val_ci, test_ci, best_params = run_random_survival_forest(
            train_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_train.csv'),
            val_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_val.csv'),
            test_path=os.path.join(DATA_DIR, 'preprocessed_clinical_data_test.csv'),
            time_col='Survival.time',
            event_col='deadstatus.event',
            variance_threshold=0.01,  # Remove features with variance < 0.01
            random_state=seed  # Use seed for RSF random state
        )

        train_ci_scores.append(train_ci)
        val_ci_scores.append(val_ci)
        test_ci_scores.append(test_ci)
        best_params_list.append(best_params)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY - RANDOM SURVIVAL FOREST")
    print(f"{'=' * 60}")
    print(f"\nTrain CI scores: Mean = {np.mean(train_ci_scores):.4f} ± {np.std(train_ci_scores):.4f}")
    print(f"Val CI scores: Mean = {np.mean(val_ci_scores):.4f} ± {np.std(val_ci_scores):.4f}")
    print(f"Test CI scores: Mean = {np.mean(test_ci_scores):.4f} ± {np.std(test_ci_scores):.4f}")

    # Analyze most frequently selected hyperparameters
    print(f"\nMost frequently selected hyperparameters:")

    # Count n_estimators
    n_estimators_values = [p['n_estimators'] for p in best_params_list]
    unique_n_est, counts_n_est = np.unique(n_estimators_values, return_counts=True)
    print(f"\nn_estimators:")
    for val, count in sorted(zip(unique_n_est, counts_n_est), key=lambda x: x[1], reverse=True):
        print(f"  {val}: selected {count} times ({count / 50 * 100:.1f}%)")

    # Count max_depth
    max_depth_values = [p['max_depth'] for p in best_params_list]
    # Convert None to string for counting
    max_depth_values_str = ['None' if x is None else x for x in max_depth_values]
    unique_depth, counts_depth = np.unique(max_depth_values_str, return_counts=True)
    print(f"\nmax_depth:")
    for val, count in sorted(zip(unique_depth, counts_depth), key=lambda x: x[1], reverse=True):
        print(f"  {val}: selected {count} times ({count / 50 * 100:.1f}%)")

    # Count min_samples_split
    min_samples_values = [p['min_samples_split'] for p in best_params_list]
    unique_samples, counts_samples = np.unique(min_samples_values, return_counts=True)
    print(f"\nmin_samples_split:")
    for val, count in sorted(zip(unique_samples, counts_samples), key=lambda x: x[1], reverse=True):
        print(f"  {val}: selected {count} times ({count / 50 * 100:.1f}%)")