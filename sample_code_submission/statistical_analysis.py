import math
import numpy as np
from scipy import integrate
from scipy.stats import poisson, norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import os
from numpy import NaN
import numpy as np
import pandas as pd


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters
    """

    def n_bins(score, poids, c_alpha, result_fonc):
        ns, nb, proS, proB, poidsS, poidsB = result_fonc
        #  Définition des constantes utiles tout au long du programme
        nombre_intervaux = len(proS)  # découpage de l'intervalle de score
        # longueur de chaque intervalle de score
        long_interv = 1 / (nombre_intervaux - 1)
        # array de boolées du score:

        # création des histogrammes avec le poids
        # hist = np.histogram(, bins=(0,1,nombre_intervaux)
        #                      weights=)[0]

        nb_tot = np.histogram(score, nombre_intervaux,
                              weights=weight)[0]

        ind_max = int(c_alpha / long_interv)
        n_sign = np.zeros(len(nb_tot))
        n_bruit = np.zeros(len(nb_tot))
        for i in range(len(nb_tot)):
            if i > ind_max:
                n_sign[i] = nb_tot[i]
            else:
                n_bruit[i] = nb_tot[i]
        # nombres de signaux dans chaque bin

        n_sign = n_sign
        n_bruit = n_bruit

        return np.round(n_sign + n_bruit).astype(int)

    def likely(liste_n, bin_n_S, bin_n_B, pS, pB, W_S, W_B, mu):
        likel = 0
        NS = bin_n_S.sum()
        WS = W_S.sum()
        NB = bin_n_B.sum()
        WB = W_B.sum()
        # print(mu*NS*pS[5] + NB*pB[5])
        # print(liste_n)
        # ll = -2*sum([np.log(poisson(mu*NS*pS[i] + NB*pB[i]).pmf(liste_n))
        #             for i in range(0, np.size(liste_n))])
        # print(ll)
        # return ll
        for i in range(len(liste_n)):
            lam = mu * NS * pS[i] + NB * pB[i]
            if lam != 0:
                likel += -lam + liste_n[i] * math.log(lam)
        return likel

    def minimisateur(liste_n, liste_nS, liste_nB, wS, wB):
        EPS = 0.0001  # trick to avoid potential division by zero during the minimization
        # Forbids parameter values to be negative, so mu>EPS here.
        par_bnds = (EPS, None)
        return minimize(
            lambda mu: -2 * likely(liste_n, liste_nS, liste_nB, pS, pB, wS, wB, mu),
            0.5,
            bounds=par_bnds[1],
        )

    def incertitude(liste_n, liste_nS, liste_nB, pS, pB, wS, wB):

        mini = minimisateur(liste_n, liste_nS, liste_nB, wS, wB).x[0]
        print(f"L'estimateur de mu chapeau vaut : {mini}")

        mu_axis_values = np.linspace(mini-1, mini+1, 2000)

        loglike_values = np.array([-2*likely(liste_n, liste_nS, liste_nB, pS, pB, wS, wB, mu)
            for mu in mu_axis_values]).flatten()

        plt.plot(mu_axis_values, loglike_values - min(loglike_values), label="log-likelihood")
        plt.hlines(1, min(mu_axis_values), max(mu_axis_values), linestyle="--", color="tab:gray")

        # This is the code to search for which mu values the log-likelihood ratio takes
        # the value 1. For this we pick up first the indexes:
        idx = np.argwhere(np.diff(np.sign(loglike_values - min(loglike_values) - 1))).flatten()

        # idx = np.append(idx, 50)

        # and we plot then the position of the mu values:
        plt.plot(mu_axis_values[idx], [1, 1], "ko", label=r"$1\sigma$ interval")
        plt.plot(mu_axis_values[idx[0]] * np.ones(2), [0, 1], "k--")
        plt.plot(mu_axis_values[idx[1]] * np.ones(2), [0, 1], "k--")
        plt.xlabel(r"$\mu$")
        plt.ylabel(r"$-2\log {\cal L}(\mu)/{\cal L}(\hat{\mu})$")
        plt.title(r"Log-likelihood profile with respect to $\mu$")

        # If the log-likelihood ratio is parabolic, sigma_mu is just the half of the
        # difference between the 2 intersection points of the likelihood ratio with 1.
        sigma_mu = np.diff(mu_axis_values[idx]) / 2

        plt.plot(mu_axis_values, ((mu_axis_values-mini)/sigma_mu)**2, linestyle='-.',
                 color='tab:gray', label='parabola approximation')
        plt.legend(facecolor='w')
        plt.show()

        return mini, abs(mini - mu_axis_values[idx[0]])

    train_set = saved_info[0]

    nS, nB, pS, pB, wS, wB = saved_info[1]

    # print(nS, nB)

    liste_n = n_bins(score, train_set["weights"], 0.5, (nS, nB, pS, pB, wS, wB))

    mu_hat, del_mu_stat = incertitude(liste_n, nS, nB, pS, pB, wS, wB)

    return {
        "mu_hat": mu_hat,
        "del_mu_stat": del_mu_stat,
        # "del_mu_sys": del_mu_sys,
        # "del_mu_tot": del_mu_tot,
    }


# train_set dictionnaire avec "data", "labels" et "weights"
def calculate_saved_info(model, train_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(train_set["data"])

    def Proba_hist(n, model, train_set):
        """
        Création de deux tableaux : Pb et Ps qui contiendront la probabilité d'avoir du background
        et du signal respectivement dans chaque Bin
        """
        # score = np.array([random() for i in range(len(train_set["data"]))])

        # We initialize the probability of an event being a signal or background one to 0.
        # nS = np.zeros(n+1)
        # nB = np.zeros(n+1)
        wS = np.zeros(n)
        wB = np.zeros(n)

        for i in range(len(score)):

            if train_set["labels"][i] == 1:
                wS[round((n - 1) * score[i])] += train_set["weights"][i]
            else:
                wB[round((n - 1) * score[i])] += train_set["weights"][i]

        S = train_set["weights"] * (train_set["labels"])
        B = train_set["weights"] * (1 - train_set["labels"])

        nS, bin = np.histogram(score, n, weights=S)
        nB, bin2 = np.histogram(score, n, weights=B)

        pS, bin = np.histogram(score, n, weights=S, density=True)
        pS = pS * (bin[1] - bin[0])
        pB, bin2 = np.histogram(score, n, weights=B, density=True)
        pB = pB * (bin[1] - bin[0])

        def histo_plot(score: list, weight: list, label: list, nbins):
            """
            fonction permettant d'afficher les histogrammes
            """
            dataTot = np.array([[score[i], weight[i], label[i]] for i in range(len(score))])
            dataS = dataTot[dataTot[:, 2] == 1]
            dataB = dataTot[dataTot[:, 2] == 0]
            plt.hist(
                dataS[:, 0],
                density=True,
                weights=dataS[:, 1],
                bins=nbins,
                range=(0, 1),
                color="red",
                alpha=0.5,
                label="Signal",
            )
            plt.hist(
                dataB[:, 0],
                density=True,
                weights=dataB[:, 1],
                bins=nbins,
                range=(0, 1),
                color="blue",
                alpha=0.5,
                label="Noise",
            )
            plt.xlabel("score")
            plt.ylabel("number of occurence")
            plt.title("Distribution of the score for signal S and noise B")
            plt.legend()
            plt.show()

        histo_plot(score, train_set["weights"], train_set["labels"], n)
        return nS, nB, pS, pB, wS, wB

    Proba_et_Poids = Proba_hist(25, model, train_set)

    return train_set, Proba_et_Poids


# print(compute_mu([0.2, 0.3, 0.4, 0.5, 0.8], [1, 1, 1, 1, 1],
#      {'gamma': 60, 'beta': 1000}))
