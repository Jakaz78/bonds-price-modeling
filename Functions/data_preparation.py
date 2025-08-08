import re
import pandas as pd
import numpy as np

def sanitize_name(name):
    """Clean variable names for safe processing"""
    name = re.sub(r'[^0-9a-zA-Z_]+', '_', str(name))
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def sanitize_columns(df):
    """Apply name sanitization to all DataFrame columns"""
    df = df.copy()
    df.columns = [sanitize_name(c) for c in df.columns]
    return df


def load_and_prepare_data(filepath):
    """Load data and split into training/test sets"""
    data = pd.read_excel(filepath)
    data = data.iloc[:, 1:].copy()  # Remove first column (date)
    data = sanitize_columns(data)
    alpha = 0.05

    # Linear interpolation for missing values
    for col in data.columns:
        data[col] = data[col].interpolate(method='linear', limit_direction='both')

    # 80/20 train-test split
    n = len(data)
    train_size = int(np.floor(0.8 * n))
    data_learning = data.iloc[:train_size].copy()
    data_test = data.iloc[train_size:].copy()
    return data_learning, data_test, alpha


def filter_by_correlation_with_y(data, y_name="CLOSE", low_thr=0.3, high_thr=0.75):
    """Remove variables with correlation outside acceptable range [low_thr, high_thr]"""
    if y_name not in data.columns:
        raise ValueError(f"Column {y_name} not found in data.")

    explanatory_vars = [col for col in data.columns if col != y_name]
    to_remove = []

    print(f"Checking correlations with {y_name} (threshold: |r| ∈ [{low_thr}, {high_thr}]):")

    for var in explanatory_vars:
        r = abs(data[var].corr(data[y_name]))

        if r > high_thr or r < low_thr:
            to_remove.append(var)
            reason = f"|r| = {r:.4f} > {high_thr}" if r > high_thr else f"|r| = {r:.4f} < {low_thr}"
            status = "REMOVED"
        else:
            reason = f"|r| = {r:.4f} ∈ [{low_thr}, {high_thr}]"
            status = "KEPT"

        print(f"- {var}: r = {r:.4f} -> {status} ({reason})")

    print(f"\nVariables removed due to correlation outside [{low_thr}, {high_thr}]: {to_remove}")

    result = data.drop(columns=to_remove)
    return result, to_remove


def remove_inflation_variable(data_learning, data_test):
    """Remove INFLATION variable before log transformation"""
    for df in [data_learning, data_test]:
        if 'INFLATION' in df.columns:
            df.drop(columns=['INFLATION'], inplace=True)
    print("Removed INFLATION variable before log transformation")
    return data_learning, data_test


def log_transform(df):
    """Apply logarithmic transformation to data"""
    df_copy = df.copy()

    mask_nonpos = (df_copy <= 0).any(axis=1)
    if mask_nonpos.any():
        print(f"Warning: {mask_nonpos.sum()} rows contain values <= 0")
        df_copy = df_copy.where(df_copy > 0)
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].interpolate(method='linear', limit_direction='both')

    df_copy = df_copy.dropna()
    result = np.log(df_copy)
    print("Data successfully log-transformed")
    return result


def remove_low_variance_variables(data_stationary, data_test_stationary):
    """Remove variables with very low variance"""
    print("Coefficient of variation before removal:")
    for col in data_stationary.columns:
        cv = data_stationary[col].std() / abs(data_stationary[col].mean()) * 100
        var = data_stationary[col].var()
        print(f"{col} - CV: {cv:.2f}%, Variance: {var:.6f}")

    to_remove = []

    if to_remove:
        data_stationary = data_stationary.drop(columns=to_remove)
        data_test_stationary = data_test_stationary.drop(columns=to_remove, errors='ignore')
        print(f"\nRemoved low variance variables: {to_remove}")

    print("\nCoefficient of variation after removal:")
    for col in data_stationary.columns:
        cv = data_stationary[col].std() / abs(data_stationary[col].mean()) * 100
        var = data_stationary[col].var()
        print(f"{col} - CV: {cv:.2f}%, Variance: {var:.6f}")

    return data_stationary, data_test_stationary
