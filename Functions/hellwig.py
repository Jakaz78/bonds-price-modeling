import pandas as pd
from itertools import combinations
import numpy as np

def hellwig_method_original(y, X):
    """
    Hellwig's method for variable selection
    h_kj = r0j^2 / (1 + sum |correlations with others in combination|)
    H_k = sum h_kj
    """
    all_vars = list(X.columns)

    combined_data = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    R = combined_data.corr().values

    r0 = R[1:, 0]  # correlations of X variables with y
    Rxx = R[1:, 1:]  # correlation matrix between X variables

    results = []

    for k in range(1, len(all_vars) + 1):
        for combo in combinations(range(len(all_vars)), k):
            indices = list(combo)
            r0_sub = r0[indices]
            Rxx_sub = Rxx[np.ix_(indices, indices)]

            h_kj = []
            for j in range(len(indices)):
                if len(indices) == 1:
                    denom = 1.0
                else:
                    other_correlations = Rxx_sub[j, [i for i in range(len(indices)) if i != j]]
                    denom = 1.0 + np.sum(np.abs(other_correlations))

                h_kj.append((r0_sub[j] ** 2) / denom)

            H_k = np.sum(h_kj)
            combo_vars = [all_vars[i] for i in indices]
            results.append({
                'variables': ', '.join(combo_vars),
                'capacity': H_k,
                'var_list': combo_vars
            })

    results.sort(key=lambda x: x['capacity'], reverse=True)
    return results
