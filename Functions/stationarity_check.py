import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

def check_stationarity(series, alpha=0.05):
    """
    Check stationarity using ADF and KPSS tests
    Series is stationary when: (ADF p-value < alpha) AND (KPSS p-value > alpha)
    """
    series = pd.Series(series).dropna()
    if len(series) < 20:
        return False

    try:
        adf_result = adfuller(series, autolag='AIC')
        adf_p_value = adf_result[1]
        adf_test_result = adf_p_value < alpha
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The test statistic is outside of the range")
            kpss_result = kpss(series, nlags="auto")
            kpss_p_value = kpss_result[1]
            kpss_test_result = kpss_p_value > alpha

        is_stationary = adf_test_result and kpss_test_result
        return is_stationary
    except Exception as e:
        print(f"Error in stationarity test for {series.name}: {e}")
        return False


def analyze_stationarity(data):
    """Analyze stationarity for all variables"""
    results = {}
    non_stationary_vars = []
    stationary_vars = []

    for col in data.columns:
        is_stationary = check_stationarity(data[col])
        results[col] = "Stationary" if is_stationary else "Non-stationary"

        if is_stationary:
            stationary_vars.append(col)
        else:
            non_stationary_vars.append(col)

    print("Stationarity check results:")
    for var, status in results.items():
        print(f"- {var}: {status}")

    return non_stationary_vars, stationary_vars


def remove_nonstationarity(data, non_stationary_vars, max_diff=2):
    """Remove non-stationarity through differencing"""
    transformed_data = {}
    diff_info = {}

    # Copy stationary variables unchanged
    for col in data.columns:
        if col not in non_stationary_vars:
            diff_info[col] = {'order': 0, 'name': col}
            transformed_data[col] = data[col]

    # Difference non-stationary variables
    for var_name in non_stationary_vars:
        original_series = data[var_name]
        order = 0

        for i in range(1, max_diff + 1):
            test_series = original_series.diff(i).dropna()
            if check_stationarity(test_series):
                order = i
                current_series = original_series.diff(i)
                break

        if order == 0:
            order = max_diff
            current_series = original_series.diff(max_diff)

        new_name = f"D{order}_{var_name}" if order > 1 else f"D_{var_name}"
        diff_info[var_name] = {'order': order, 'name': new_name}
        transformed_data[new_name] = current_series

    result_df = pd.DataFrame(transformed_data)
    result_df = result_df.dropna()

    return result_df, diff_info


def apply_diff_to_test_data(data_test, diff_info):
    """Apply same differencing transformations to test data"""
    transformed_test = {}

    for original_var, info in diff_info.items():
        order = info['order']
        new_name = info['name']

        if original_var in data_test.columns:
            if order == 0:
                transformed_test[new_name] = data_test[original_var]
            else:
                transformed_test[new_name] = data_test[original_var].diff(order)

    result_df = pd.DataFrame(transformed_test)
    result_df = result_df.dropna()
    return result_df

