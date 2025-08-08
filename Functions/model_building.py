import statsmodels.api as sm

def build_ols_model(data_stationary, response_var='D_CLOSE', predictor_vars=None):
    """Build OLS regression model"""
    if predictor_vars is None:
        predictor_vars = [col for col in data_stationary.columns if col != response_var]

    Y = data_stationary[response_var]
    X = data_stationary[predictor_vars]
    X_with_const = sm.add_constant(X)

    model = sm.OLS(Y, X_with_const).fit()
    print(model.summary())

    return model, X_with_const
