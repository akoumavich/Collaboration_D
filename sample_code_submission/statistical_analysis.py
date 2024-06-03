import numpy as np
from HiggsML.systematics import systematics
import scipy
import scipy.stats as st
from IPython.display import set_matplotlib_formats, display
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from numpy import NaN

# progressbar


def progressbar(value, max=100):
    """
    function that make a progressbar, useful to see if the code is working or just crashing
    """
    from IPython.display import HTML
    return HTML("""<progress value='{value}' max='{max}' style='width: 100%'>
    {value}</progress>""".format(value=value, max=max))


def pbinit(kmin, kmax):
    return display(progressbar(kmin, kmax), display_id=True)


# computation from data

def S(sample):
    """
    function that isolate the signal from the sample
    """
    pass


def B(sample):
    """
    function that isolate the signal from the sample
    """
    pass

# computation


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.1 * mu)
    del_mu_tot = (1 / 2) * np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


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


# function computing the likelihood

# stats_law from Statistical functions (scipy.stats)
def likelihood(mu: float, model: function, n: int, stat_law: scipy.stats):
    """
    function computing the likelihood
    input: the function; mu; the model; n an interger and the statistical law
    output: function
    """
    return stat_law.pmf(n, model(mu))


def log_likelihood(function: function, mu: float, model: function, n: int, stat_law: scipy.stats):
    """
    function computing the log likelihood
    input: the function; mu; the model; n an interger and the statistical law
    output: fuction
    """
    llh = likelihood(function, mu, model, n, stat_law)
    return -2*np.log(llh)

# model


def model(mu: float, S: int, B: int):
    """
    function computing mu according to the model see in class
    input: mu; S and B
    output : float the result
    """
    return mu*S+B


def histo_mu(model: function, sample, stats_law: scipy.stats):
    """
    function to see the distribution of mu estimator according to the sample
    """
    nsim = len(sample)

    S_sample = S(sample)
    B_sample = B(sample)
    n_sample = S_sample + B_sample

    mu_sample = np.zeros([nsim, 1])
    pb = pbinit(0, nsim)
    for idx, n in np.ndenumerate(n_sample):
        pb.update(progressbar(idx[0], nsim))
        res = minimize(lambda mu: log_likelihood(mu, model, n, stats_law), .5)
        mu_sample[idx] = res.x[0]

    fig, fig_axes = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches(w=14, h=3)
    fig_axes[0].hist(mu_sample, density=True)
    fig_axes[0].set_xlabel(r'$\mu$')
    fig_axes[0].set_title(r'Fitted $\mu$' + '\n distribution')
    fig_axes[1].plot(S_sample, mu_sample, 'k+')
    fig_axes[1].set_xlabel('$S$')
    fig_axes[1].set_ylabel(r'$\mu$')
    fig_axes[1].set_title(r'Fitted $\mu$' + '\n vs. true $S$ in data')
    fig_axes[2].plot(B_sample, mu_sample, 'k+')
    fig_axes[2].set_xlabel('$B$')
    fig_axes[2].set_ylabel(r'$\mu$')
    fig_axes[2].set_title(r'Fitted $\mu$' + '\n vs. true $B$ in data')
    plt.show()


def loglik_estimation(model: function, sample, stats_law: scipy.stats):
    """
    function that compute the loglikelihood estimation
    """
    pass


def binned_shape_loglikelihood(model: function, sample, stats_law: scipy.stats):
    """
    function that compute the binned shape loglikelihood estimation
    """
    pass


def profile_loglikelihood(model: function, sample, stats_law: scipy.stats):
    """
    function that compute the binned shape loglikelihood estimation
    """
    pass
