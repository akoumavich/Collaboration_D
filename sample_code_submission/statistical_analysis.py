import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import os
from numpy import NaN
import numpy as np
import pandas as pd

os.chdir('D:/cours/centralesupelec/1A/ST 4/ST4 black swan/EI/Collaboration_D')


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters
    """

    # score = score.flatten() > 0.5
    # score = score.astype(int)

    # mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    # del_mu_stat = (
    #     np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    # )
    # del_mu_sys = abs(0.1 * mu)
    # del_mu_tot = (1 / 2) * np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    # tentative d'histogramme
    n_sample = saved_info["gamma"] + saved_info["beta"]

    def model(mu):
        return mu*saved_info["gamma"]+saved_info["beta"]

    def loglikelihood(mu, n):
        return -2*np.log(poisson.pmf(n, model(mu)))

    res = minimize(lambda mu: loglikelihood(mu, n_sample), .5)
    mu = res["x"][0]

    return {
        "mu_hat": mu,
        # "del_mu_stat": del_mu_stat,
        # "del_mu_sys": del_mu_sys,
        # "del_mu_tot": del_mu_tot,
    }


print(compute_mu([0.2, 0.3, 0.4, 0.5, 0.8], [1, 1, 1, 1, 1],
                 {'gamma': 60, 'beta': 1000}))


def calculate_saved_info(model, train_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    # train_plus_syst = systematics(
    #     data_set=train_set,
    #     tes=1.03,
    #     jes=1.03,
    #     soft_met=1.0,
    #     seed=31415,
    #     w_scale=None,
    #     bkg_scale=None,
    #     verbose=0,
    # )

    score = model.predict(train_set["data"])

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    print("score shape after threshold", score.shape)

    gamma = np.sum(train_set["weights"] * score)

    beta = np.sum(train_set["weights"] * (1 - score))

    saved_info = {"beta": beta, "gamma": gamma}

    print("saved_info", saved_info)

    return saved_info
