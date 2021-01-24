import numpy as np
import statsmodels.api as sm


def fit(stim, k_len, spike_counts):
    X_stim = stack_shifted(stim, k_len)

    X_stim_with_const = sm.add_constant(X_stim, prepend=False)

    # fit GLM
    glm_model = sm.GLM(spike_counts, X_stim_with_const, family=sm.families.Poisson())
    glm_results = glm_model.fit()
    # glm_results.summary()

    k_est = glm_results.params[:-1]
    dc_est = glm_results.params[-1]

    return k_est, dc_est, glm_results


def stack_shifted(x, num_shift):
    len_x, dim_x = x.shape
    x_shifted = []

    for j in reversed(range(num_shift)):  # for each time shift  # reverse to put past to the left!
        pad = np.zeros((j, dim_x))
        xs = x[0:len_x-j, :]
        x_padded = np.row_stack([pad, xs])
        #print(pad, xs, x_padded)
        # print(x_padded)

        x_shifted.append(x_padded)

    # x_shifted  = list of time-shifted x's
    return np.column_stack(x_shifted)   # return as a numpy array

