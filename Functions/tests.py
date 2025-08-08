from statsmodels.stats.diagnostic import (het_breuschpagan, het_goldfeldquandt,acorr_ljungbox, acorr_breusch_godfrey)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, jarque_bera

def test_normality_of_residuals(residuals):
    """Test normality of residuals using Shapiro-Wilk and Jarque-Bera tests"""
    W, p_shapiro = shapiro(residuals)
    print(f"Shapiro-Wilk Test:")
    print(f" Statistic W = {W:.4f}")
    print(f" p-value = {p_shapiro:.4f}")
    if p_shapiro < 0.05:
        print(" Conclusion: Reject H0 - residuals are not normal")
    else:
        print(" Conclusion: Cannot reject H0 - residuals are normal")

    jb_stat, jb_p = jarque_bera(residuals)
    print(f"\nJarque-Bera Test:")
    print(f" Statistic JB = {jb_stat:.4f}")
    print(f" p-value = {jb_p:.4f}")
    if jb_p < 0.05:
        print(" Conclusion: Reject H0 - residuals are not normal")
    else:
        print(" Conclusion: Cannot reject H0 - residuals are normal")

    return W, p_shapiro, jb_stat, jb_p


def test_autocorrelation_comprehensive(residuals, model):
    """Comprehensive autocorrelation tests"""
    print("Autocorrelation tests:")

    dw_stat = durbin_watson(residuals)
    print(f"\nDurbin-Watson Test:")
    print(f" Statistic DW = {dw_stat:.4f}")
    if 1.5 <= dw_stat <= 2.5:
        print(" Conclusion: Cannot reject H0 - no autocorrelation")
    else:
        print(" Conclusion: Suspected autocorrelation")

    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_stat = lb_result['lb_stat'].iloc[0]
    lb_p = lb_result['lb_pvalue'].iloc[0]
    print(f"\nLjung-Box Test:")
    print(f" Statistic LB = {lb_stat:.4f}")
    print(f" p-value = {lb_p:.4f}")
    if lb_p > 0.05:
        print(" Conclusion: Cannot reject H0 - no autocorrelation")
    else:
        print(" Conclusion: Reject H0 - autocorrelation present")

    bg_stat, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=2)
    print(f"\nBreusch-Godfrey Test:")
    print(f" Statistic LM = {bg_stat:.4f}")
    print(f" p-value = {bg_p:.4f}")
    if bg_p > 0.05:
        print(" Conclusion: Cannot reject H0 - no autocorrelation")
    else:
        print(" Conclusion: Reject H0 - autocorrelation present")

    return dw_stat, lb_stat, lb_p, bg_stat, bg_p


def test_heteroskedasticity_comprehensive(residuals, X_with_const, fitted_values):
    """Comprehensive heteroskedasticity tests"""
    print("Heteroskedasticity tests:")

    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"\nBreusch-Pagan Test:")
    print(f" Statistic BP = {bp_stat:.4f}")
    print(f" p-value = {bp_p:.4f}")
    if bp_p > 0.05:
        print(" Conclusion: Cannot reject H0 - homoskedasticity")
    else:
        print(" Conclusion: Reject H0 - heteroskedasticity")

    gq_stat, gq_p, _ = het_goldfeldquandt(residuals, X_with_const)
    print(f"\nGoldfeld-Quandt Test:")
    print(f" Statistic GQ = {gq_stat:.4f}")
    print(f" p-value = {gq_p:.4f}")
    if gq_p > 0.05:
        print(" Conclusion: Cannot reject H0 - homoskedasticity")
    else:
        print(" Conclusion: Reject H0 - heteroskedasticity")

    return bp_stat, bp_p, gq_stat, gq_p


def test_multicollinearity(X_with_const):
    """Test multicollinearity using Variance Inflation Factor (VIF)"""
    print("Multicollinearity test (VIF):")

    if X_with_const.shape[1] <= 2:
        print("Only one explanatory variable - no multicollinearity issues")
        return [1.0]

    vif_values = []
    for i in range(X_with_const.shape[1]):
        if X_with_const.columns[i] != 'const':
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_values.append(vif)
            print(f" VIF for {X_with_const.columns[i]}: {vif:.4f}")

    max_vif = max(vif_values) if vif_values else 1.0
    print(f"Maximum VIF: {max_vif:.4f}")

    if max_vif < 5:
        print("Conclusion: No multicollinearity issues")
    elif max_vif < 10:
        print("Conclusion: Moderate multicollinearity")
    else:
        print("Conclusion: Severe multicollinearity")

    return vif_values
