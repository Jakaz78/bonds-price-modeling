# econometric_project.py

import warnings

warnings.filterwarnings("ignore")
from Functions.data_preparation import *
from Functions.stationarity_check import *
from Functions.tests import *
from Functions.model_building import *
from Functions.hellwig import *
from Functions.plots_creation import *
import statsmodels.api as sm



def main():
    print("=== ECONOMETRIC PROJECT ===\n")

    # 1) Data loading
    print("1. Loading data...")
    data_learning, data_test, alpha = load_and_prepare_data("data.xlsx")
    print(f"Data loaded: {data_learning.shape[0]} training obs., {data_test.shape[0]} test obs.")
    print(f"Variables: {list(data_learning.columns)}")

    # 2) Correlation filtering
    print("\n2. Filtering variables by correlation with target...")
    data_filtered, removed_vars = filter_by_correlation_with_y(data_learning)
    data_test = data_test.drop(columns=[v for v in removed_vars if v in data_test.columns], errors='ignore')

    # 3) Remove INFLATION
    print("\n3. Removing INFLATION variable...")
    data_filtered, data_test = remove_inflation_variable(data_filtered, data_test)

    # 4) Log transformation
    print("\n4. Log transformation...")
    data_log = log_transform(data_filtered)
    data_test_log = log_transform(data_test[data_log.columns])

    print("\nCreating correlation heatmap...")
    plot_correlation_heatmap(data_filtered)
    # 5) Stationarity analysis
    print("\n5. Stationarity analysis...")
    non_stationary_vars, stationary_vars = analyze_stationarity(data_log)

    # 6) Remove non-stationarity
    print("\n6. Removing non-stationarity through differencing...")
    data_stationary, diff_info = remove_nonstationarity(data_log, non_stationary_vars, max_diff=2)
    data_test_stationary = apply_diff_to_test_data(data_test_log, diff_info)

    # 7) Re-check stationarity
    print("\n7. Re-checking stationarity...")
    _, _ = analyze_stationarity(data_stationary)

    # 8) Remove low variance variables
    print("\n8. Removing low variance variables...")
    data_stationary, data_test_stationary = remove_low_variance_variables(data_stationary, data_test_stationary)

    # 9) Hellwig method
    print("\n9. Hellwig method - variable selection...")
    if 'D_CLOSE' not in data_stationary.columns:
        raise ValueError("D_CLOSE not found after transformations!")

    Y = data_stationary['D_CLOSE']
    X = data_stationary.drop(columns=['D_CLOSE'])

    hellwig_results = hellwig_method_original(Y, X)

    print("Best variable combinations (Hellwig method):")
    for i, result in enumerate(hellwig_results[:5], 1):
        print(f"{i}. {result['variables']} -> Capacity: {result['capacity']:.4f}")

    best_combination = hellwig_results[0]
    best_vars = best_combination['var_list']
    print(f"\nSelected variables for model: {best_vars}")
    print(f"Hellwig capacity: {best_combination['capacity']:.4f}")

    # 10) Prepare final data
    final_columns = ['D_CLOSE'] + best_vars
    data_final = data_stationary[final_columns].copy()
    data_test_final = data_test_stationary[final_columns].copy()

    # 11) Build model
    print("\n10. Building econometric model...")
    model, X_with_const = build_ols_model(data_final, 'D_CLOSE', best_vars)

    # 12) Model diagnostics
    residuals = model.resid
    fitted_values = model.fittedvalues

    print("\n11. Model diagnostics:")

    print("\n--- Normality test ---")
    test_normality_of_residuals(residuals)

    print("\n--- Autocorrelation tests ---")
    test_autocorrelation_comprehensive(residuals, model)

    print("\n--- Heteroskedasticity tests ---")
    test_heteroskedasticity_comprehensive(residuals, X_with_const, fitted_values)

    print("\n--- Multicollinearity test ---")
    test_multicollinearity(X_with_const)

    # 13) Parameter interpretation
    print("\n12. Parameter interpretation:")
    for param_name, coef in model.params.items():
        p_value = model.pvalues[param_name]
        t_value = model.tvalues[param_name]

        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        elif p_value < 0.1:
            significance = "."

        print(f"\n{param_name}:")
        print(f"  Coefficient: {coef:.6f}")
        print(f"  t-statistic: {t_value:.4f}")
        print(f"  p-value: {p_value:.4f} {significance}")

        if p_value < 0.05:
            print(f"  Status: STATISTICALLY SIGNIFICANT")
        else:
            print(f"  Status: NOT STATISTICALLY SIGNIFICANT")

    # 14) Out-of-sample evaluation
    print("\n13. Out-of-sample model evaluation...")

    test_predictors = X_with_const.columns
    missing_in_test = [col for col in test_predictors if col not in data_test_final.columns and col != 'const']

    if missing_in_test:
        print(f"Warning: Missing variables in test data: {missing_in_test}")

    X_test = data_test_final[best_vars]
    X_test_with_const = sm.add_constant(X_test)
    Y_test = data_test_final['D_CLOSE']

    predictions = model.predict(X_test_with_const)

    mae = np.mean(np.abs(Y_test - predictions))
    rmse = np.sqrt(np.mean((Y_test - predictions) ** 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((Y_test - predictions) / Y_test) * 100
        mape_values = mape_values[np.isfinite(mape_values)]
        mape = np.mean(mape_values) if len(mape_values) > 0 else np.nan

        smape = np.mean(2 * np.abs(Y_test - predictions) / (np.abs(Y_test) + np.abs(predictions))) * 100

    print(f"\n=== OUT-OF-SAMPLE RESULTS (Test Set) ===")
    print(f"MAE = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    if not np.isnan(mape):
        print(f"MAPE = {mape:.2f}%")
    else:
        print("MAPE = NaN (very small D_CLOSE values)")
    print(f"sMAPE = {smape:.2f}%")

    print("\nCreating actual vs predicted plot...")
    plot_actual_vs_predicted(Y_test, predictions)

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
