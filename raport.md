# Econometric Analysis Report: Financial Time Series Modeling

## Table of Contents
- [Executive Summary](#executive-summary)
- [Data and Methodology](#data-and-methodology)
- [Results](#results)
- [Model Diagnostics](#model-diagnostics)
- [Performance](#Out-of-Sample-Performance)

## Executive Summary

This report presents a comprehensive econometric analysis of financial time series data using Python. The analysis includes variable selection through correlation filtering, stationarity testing, model building using Hellwig's method, and comprehensive diagnostic testing.

**Key Results:**
- Final model R² = 0.138
- Selected variables: D_XAUUSD, D_WIG20, D_OIL
- Out-of-sample RMSE: 0.111

## Data and Methodology

**Primary Source:** [www.stooq.com](https://www.stooq.com)

### Dependent Variable
**CLOSE** - Yield on 10-year Polish government bonds

### Explanatory Variables

| Variable | Full Name | Description |
|----------|-----------|-------------|
| **10YDEBOND** | German 10-Year Bond Yield | Yield on 10-year German government bonds |
| **10YUSBOND** | US 10-Year Bond Yield | Yield on 10-year US government bonds |
| **DETAL** | Retail Sales | Month-to-month retail sales growth |
| **XAUUSD** | Gold Price | Gold price in US dollars |
| **S&P500** | S&P 500 Index | ETF of 500 largest US publicly traded companies |
| **PMI** | Purchasing Managers' Index | Industrial activity indicator |
| **WIG20** | WIG20 Index | 20 largest Polish publicly traded companies |
| **OIL** | Oil Price | Crude oil price per barrel |
| **UNEMPLOYMENT** | Unemployment Rate | Unemployment rate in Poland |
| **USDPLN** | USD/PLN Exchange Rate | US dollar exchange rate expressed in Polish zloty |
| **INFLATION** | Inflation Rate | Year-over-year inflation rate |
| **WIBOR** | Warsaw Interbank Offered Rate | Reference interest rate for Polish interbank market |


### Methodology Steps

1. **Data Preprocessing**
   - Linear interpolation for missing values
   - 80/20 train-test split
   - Variable name sanitization

2. **Variable Selection**
   - Correlation-based filtering (keep |r| ∈ [0.3, 0.75])
   - Removed variables with too low/high correlation with target

3. **Stationarity Analysis**
   - ADF test (H₀: unit root exists)
   - KPSS test (H₀: series is stationary)
   - First differencing for non-stationary series

4. **Model Building**
   - Hellwig's method for optimal variable combination
   - OLS regression with selected variables

5. **Diagnostic Testing**
   - Normality tests (Shapiro-Wilk, Jarque-Bera)
   - Autocorrelation tests (Durbin-Watson, Ljung-Box)
   - Heteroskedasticity tests (Breusch-Pagan)
   - Multicollinearity check (VIF)

## Results

### Variable Selection Results

**Correlation Analysis:**
Variables removed due to correlation outside [0.3, 0.75]:

INFLATION (|r| = 0.7867 > 0.75)

10YUSBOND (|r| = 0.7928 > 0.75)

USDPLN (|r| = 0.2271 < 0.3)

WIBOR (|r| = 0.9371 > 0.75)

PMI (|r| = 0.2868 < 0.3)

DETAL (|r| = 0.0207 < 0.3)

**Variables kept:** CLOSE, XAUUSD, WIG20, S_P500, OIL

### Stationarity Analysis

All variables were found to be **non-stationary** in levels, requiring first differencing:

| Variable | Original Status | After Differencing |
|----------|----------------|-------------------|
| CLOSE | Non-stationary | D_CLOSE - Stationary |
| XAUUSD | Non-stationary | D_XAUUSD - Stationary |
| WIG20 | Non-stationary | D_WIG20 - Stationary |
| S_P500 | Non-stationary | D_S_P500 - Stationary |
| OIL | Non-stationary | D_OIL - Stationary |

### Hellwig Method Results

**Top 5 variable combinations:**

| Rank | Variables | Hellwig Capacity |
|------|-----------|-----------------|
| 1 | D_XAUUSD, D_WIG20, D_OIL | 0.0827 |
| 2 | D_XAUUSD, D_WIG20, D_S_P500, D_OIL | 0.0716 |
| 3 | D_XAUUSD, D_WIG20 | 0.0700 |
| 4 | D_XAUUSD, D_OIL | 0.0645 |
| 5 | D_XAUUSD, D_S_P500, D_OIL | 0.0613 |

**Selected combination:** D_XAUUSD, D_WIG20, D_OIL

### Model Estimation Results

**Final Model:**

D_CLOSE = -0.0059 - 0.2976D_XAUUSD - 0.2067D_WIG20 + 0.1779*D_OIL + ε


**Parameter Estimates:**

| Parameter | Coefficient | Std. Error | t-statistic | p-value | Significance |
|-----------|------------|------------|-------------|---------|--------------|
| const | -0.005945 | 0.004 | -1.558 | 0.1206 | |
| D_XAUUSD | -0.297615 | 0.082 | -3.612 | 0.0004 | *** |
| D_WIG20 | -0.206587 | 0.060 | -3.423 | 0.0007 | *** |
| D_OIL | 0.177856 | 0.041 | 4.307 | 0.0000 | *** |

**Model Fit Statistics:**
- R² = 0.138
- Adjusted R² = 0.127  
- F-statistic = 12.58 (p < 0.001)
- Observations = 240

## Model Diagnostics

### Assumption Testing Results

| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| **Normality** | | | |
| Shapiro-Wilk | W = 0.9820 | 0.0039 | Reject H₀ - Non-normal |
| Jarque-Bera | JB = 16.62 | 0.0002 | Reject H₀ - Non-normal |
| **Autocorrelation** | | | |
| Durbin-Watson | DW = 1.833 | - | No autocorrelation |
| Ljung-Box | LB = 13.996 | 0.1732 | No autocorrelation |
| Breusch-Godfrey | LM = 1.074 | 0.5847 | No autocorrelation |
| **Heteroskedasticity** | | | |
| Breusch-Pagan | BP = 5.809 | 0.1213 | Homoskedastic |
| Goldfeld-Quandt | GQ = 1.835 | 0.0006 | Heteroskedastic |
| **Multicollinearity** | | | |
| Max VIF | 1.078 | - | No issues |

### Economic Interpretation

**D_OIL (+0.178):** Positive relationship suggests that increases in oil price changes positively affect the target variable, which is economically plausible for energy-sensitive markets.

**D_XAUUSD (-0.298):** Negative relationship with gold price changes indicates a "flight to safety" effect - when gold prices rise (uncertainty increases), the target variable tends to decrease.

**D_WIG20 (-0.207):** Negative relationship with Polish stock index changes may indicate sector-specific effects or inverse market dynamics.

## Out-of-Sample Performance

**Test Set Evaluation:**
- **MAE:** 0.083
- **RMSE:** 0.111
- **MAPE:** 212.4%
- **sMAPE:** 148.3%

**Performance Assessment:**
The model shows **limited practical forecasting ability** with very high percentage errors (MAPE > 200%). While the model is statistically significant and passes most diagnostic tests, the high forecast errors suggest the model is better suited for understanding relationships rather than precise prediction.



