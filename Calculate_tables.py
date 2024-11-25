import statsmodels.api as sm
import scipy
from arch.bootstrap import MCS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import heapq
from sklearn.metrics import roc_auc_score
import arch.covariance.kernel as kernels
import numpy as np
from matplotlib.collections import QuadMesh
from sklearn.linear_model import LinearRegression
from matplotlib.text import Text
from typing import Callable, Literal, Optional, Union
from scipy.stats import t, chi2
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def gw(
    L: np.array,
    tau: int,
    H: Optional[np.array] = None,
    covar_style: str = "sample",
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Test of Equal Conditional Predictive Ability by Giacomini and White (2006).
    Used here for testing and debugging but made available through the package interface.

    This is a reimplementation from the MATLAB code provided by
    Giacomini (https://gist.github.com/ogrnz/91f37140011d1c2447934766274c4070)

    References:
        - Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. Econometrica, 74(6), 1545-1578.

    :param L: (Tx2) array of forecast losses
    :param H: (Txq) array of instruments. If `None` provided, defaults to the unconditional EPA (DM test)
    :param tau: Forecast horizon
    :param covar_style: (default 'sample')
        How to compute the covariance matrix.
        Either the sample covariance ('sample') or an HAC estimator ('hac').
    :param kernel: (default `None`)
        If multistep forecast (`tau` > 1), the covariance matrix needs to take
        into account the correlation structure (`HAC` estimator).
        Original implementation uses a Bartlett kernel with bandwidth `tau - 1`.
        If a `str`, must match one of `arch` package variance estimator:
         > https://arch.readthedocs.io/en/latest/covariance/covariance.html
        If a `Callable`, must simply return a (qxq) covariance matrix (see arg `H`).
    :param bw: (default `None`)
        Bandwidth of the `kernel`. Typically set to `forecasting horizon - 1`.
        If set to `None`, will let the kernel compute the optimal bandwidth if supported.
    :param kernel_kwargs: (default `None`)
        An optional dict of `argument: value` passed to `kernel`.
        If `kernel` is a `Callable`, the eventual bandwidth must be passed here.
    :param alpha: Significance level
    :return: tuple(S, crit_val, p_val)
        S: test statistic,
        cval: critical value for significance lvl,
        pval: p-value of test
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    T, q = L.shape[0], H.shape[1] if H is not None else 1
    d = L[:, 0] - L[:, 1]

    # Default instruments for unconditional EPA
    H = np.ones((T, 1)) if H is None else H

    reg = H * d[:, np.newaxis]

    if tau == 1:  # One-step
        beta = np.linalg.lstsq(reg, np.ones(T), rcond=None)[0][0]
        residuals = np.ones((T, 1)) - (beta * reg)
        mean_residuals = np.mean(residuals, axis=1)
        S = T * (1 - np.mean(mean_residuals**2))
    else:  # Multistep
        omega = compute_covariance(
            reg, covar_style, kernel=kernel, bw=bw, kernel_kwargs=kernel_kwargs
        )
        zbar = reg.mean().T
        S = T * zbar.T @ np.linalg.pinv(omega) @ zbar

    dof = reg.shape[1]
    cval = chi2.ppf(1 - alpha, dof)
    pval = 1 - chi2.cdf(abs(S), dof)

    return S, cval, pval


def mgw(
    L: np.array,
    H: Optional[np.array] = None,
    covar_style: Literal["sample", "hac"] = "sample",
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
    alpha: float = 0.05,
):
    """
    Implements the multivariate Giacomini-White (MGW) (Borup et al., 2022) test of equal predictive ability.

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    Notes:
        If only 2 models are compared, it reduces to the Giacomini-White test (GW) (Giacomini and White, 2006)
        If further no conditioning information H is given, it reduces to the
        original Diebold-Mariano test (DM) (Diebold and Mariano, 1995)
        If more than 2 models are compared but with no conditioning information H,
        it reduces to multivariate Diebold-Mariano (MDM) (Mariano and Preve, 2012)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjaer, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Diebold, F.X., and R.S. Mariano (1995) ‘Comparing Predictive Accuracy,’ Journal
        of Business and Economic Statistics 13, 253–263.
        - Giacomini, R., & White, H. (2006). Tests of conditional predictive ability.
        Econometrica, 74(6), 1545-1578.
        - Mariano, R. S., & Preve, D. (2012). Statistical tests for multiple forecast comparison.
        Journal of econometrics, 169(1), 123-130.

    :param L:
        Txk matrix of losses of k models with T forecasts.
    :param H: (default `None`)
        Txq matrix of a constant and information set (test function).
        If not provided, set to a (Tx1) column vector of 1, amounts to the
        unconditional MWG test, which is equivalent to the multivariate Diebold-Mariano (Mariano and Preve, 2012).
    :param covar_style: (default 'sample')
        How to compute the covariance matrix.
        Either the sample covariance ('sample') or an HAC estimator ('hac').
    :param kernel: (default `None`)
        If covariance matrix is an HAC estimator, what type to compute.
        If a `str`, must match one of `arch` package variance estimator:
         > https://arch.readthedocs.io/en/latest/covariance/covariance.html
        If a `Callable`, must simply return a
    :param bw: (default `None`)
        Bandwidth of the `kernel`. Typically set to `forecasting horizon - 1`.
        If set to `None`, will let the kernel compute the optimal bandwidth if supported.
    :param kernel_kwargs: (default `None`)
        An optional dict of `argument: value` passed to `kernel`.
        If `kernel` is a `Callable`, the eventual bandwidth must be passed here.
    :param alpha: (default 0.05)
        Significance level.
    :return: tuple(S, cval, pval)
        S: float, the computed test statistic
        cval: float, the corresponding critical value
        pval: float, the p-value of S.
    """
    validate_args(L, covar_style, kernel, bw)

    if kernel_kwargs is None:
        kernel_kwargs = {}

    T, p = L.shape[0], L.shape[1] - 1
    H = np.ones((T, 1)) if H is None else H  # default to unconditional EPA
    D = np.diff(L, axis=1)
    reg = np.array([np.kron(h, d) for h, d in zip(H, D)])

    Dbar = np.mean(reg, axis=0)
    omega = compute_covariance(
        reg, covar_style, Dbar=Dbar, kernel=kernel, bw=bw, kernel_kwargs=kernel_kwargs
    )

    dof = H.shape[1] * p
    S = (T * Dbar @ np.linalg.pinv(omega) @ Dbar.T).item()
    cval = chi2.ppf(1 - alpha, dof)
    pval = 1 - chi2.cdf(S, dof)

    return S, cval, pval


def cmcs(L: np.array, H: Optional[np.array] = None, alpha: float = 0.05, **kwargs):
    """
    Perform the Conditional Model Confidence Set (CMCS).
    The MCS procedure from Hansen (2011) is adapted to use MGW (Borup et al., 2022)
    instead of bootstrapping the critical values. Allows to reduce an initial set of models to a
    set of models with equal (conditional) predictive ability.
    Also, allows to use conditioning information (`H`, hence the 'Conditional'),
    to get the best MCS based on expected future loss.

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjaer, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.

    :param L:
        (Txk) matrix of losses of k models with T forecasts.
    :param H: (default `None`)
        (Txq) matrix of a constant and information set (test function).
        If not provided, set to a (Tx1) column vector of 1, amounts to the
        unconditional MWG test, which is equivalent to the multivariate Diebold-Mariano (Mariano and Preve, 2012).
    :param alpha: (default 0.05)
        Significance level used in the MGW test.
    :param **kwargs: Arguments passed to `feval.mgw`. Usually define covariance estimator and such.
    :return: tuple(mcs, S, cval, pval, removed)
        mcs: (1xk) np.array where models included in the best model confidence set are noted as 1.
        S: float, the computed test statistic of the last test.
        cval: float, the corresponding critical value.
        pval: float, the p-value of S.
        removed: (1xk) np.array where a column represents an algorithm cycle.
            That way, we can see which model index was removed at which iteration.
    """
    T, k = L.shape

    H = np.ones((T, 1)) if H is None else H

    S, cval, pval = np.inf, 1, 1
    mcs = np.ones((1, k))
    removed = np.zeros((1, k))

    j = 0
    while S > cval:
        L_to_use = L[:, mcs[0].astype(bool)]

        # If only one model left in the set, exit loop
        if L_to_use.shape[1] == 1:
            break

        # Perform MGW
        S, cval, pval = mgw(L_to_use, H, alpha=alpha, **kwargs)

        # H0 still rejected, apply elimination criterion
        if S > cval:
            mcs, removed[0, j] = elim_rule(L, mcs, H)

        j += 1

    return mcs, S, cval, pval, removed


def elim_rule(L: np.array, mcs: np.array, H: Optional[np.array] = None):
    """
    Elimination rule that allows to rank losses based on expected future loss given the information set `H`.
    If `H` is a vector of constant, it amounts to ranking losses based on average loss.
    See Borup et al. (2022) and Hansen (2011).

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjaer, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.

    :param L:
        (Txk) matrix of losses of k models with T forecasts.
    :param mcs:
        (1xk) vector of current model confidence set, where the least performing model will be eliminated.
    :param H: (default `None`)
        (Txq) matrix of a constant and information set (test function).

    :return: tuple(mcs, removed)
        mcs: (1xk) np.array where models included in the best model confidence set are noted as 1.
        removed: (1xk) np.array where a column represents an algorithm cycle.
            That way, we can see which model index was removed at which iteration.
    """
    # Initialize
    k = mcs.shape[1]
    q = H.shape[1]
    new_k = np.count_nonzero(mcs)

    if L.shape[1] != k:
        raise ValueError(f"Dimensions of {L.shape[1]=} do not match {mcs.shape[1]=}.")

    #H[:,0] = np.array([0.9 * (1 - 0.9)**(t) for t in range(L.shape[0] - 1,-1,-1)])
    #L = L[np.argsort(L.mean(axis=1))]

    L_to_use = np.zeros((L.shape[0], new_k))
    curr_set = np.zeros((1, new_k))
    j = 0
    for i in range(k):  # TODO could be simplified?
        if mcs[0, i] == 1:
            L_to_use[:, j] = L[:, i]
            curr_set[0, j] = i
            j += 1

    combinations = np.arange(0, j).reshape(1, -1)  # TODO why matrix? could be vect
    L_hat = np.zeros(combinations.shape)


    # Estimate
    deltas = np.zeros((q, new_k - 1))
    for i in range(L_to_use.shape[1] - 1):
        Y_used = L_to_use[:, i + 1] - L_to_use[:, i]
        Y_used = Y_used.reshape(-1, 1)
        deltas[:, i] = (np.linalg.inv(H.T @ H) @ H.T @ Y_used).reshape(-1, )

    delta_L_hat = (deltas.T @ H[-1, :].T).reshape(-1, 1)

    # Normalize
    L_hat = np.zeros(L_hat.shape)
    L_hat[0, 0] = 1
    for i in range(L_to_use.shape[1] - 1):
        L_hat[0, i + 1] = L_hat[0, i] + delta_L_hat[i, 0]
    '''

    L_hat = np.zeros((1, new_k))
    exp = np.array([0.1 * (1 - 0.1)**(t) for t in range(L_to_use.shape[0] - 1,-1,-1)])
    for i in range(L_to_use.shape[1]):
        model = AutoReg(L_to_use[:, i], lags=1).fit()
        L_hat[0, i] = 0.8 * exp.T @ L_to_use[:, i] / np.sum(exp) + 0.2 * model.predict(start=L_to_use.shape[0], end=L_to_use.shape[0])[0]

    '''
    # Rank losses
    indx = np.argmax(L_hat)
    col = np.unique(combinations[0, indx])

    # Update mcs
    mcs[0, curr_set[0, col].astype(int)] = 0
    removed = curr_set[0, col]
    return mcs, removed


def validate_args(L, covar_style, kernel, bw):
    if kernel and covar_style == "sample":
        raise ValueError(f"{kernel=} incompatible with {covar_style=}.")
    if not kernel and covar_style == "hac":
        raise ValueError("Set `kernel` when using an HAC estimator.")
    if bw and covar_style == "sample":
        raise ValueError(f"{bw=} incompatible with {covar_style=}.")
    if L.shape[1] < 2:
        raise ValueError(f"Not enough columns for matrix of losses {L.shape[1]=}.")
    if not isinstance(L, np.ndarray):
        raise TypeError(
            f"Only np.ndarray is supported for the loss (currently {type(L)})."
        )
    if np.isnan(L).sum() > 0:
        raise ValueError("Ensure there are no NaN in your loss matrix.")


def compute_covariance(
    reg: np.array,
    covar_style: str,
    Dbar: Optional[np.array] = None,
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
) -> np.array:
    """
    Compute the covariance matrix omega for the given regression residuals and kernel.

    :param reg: Residuals from the regression.
    :param Dbar: Mean of the regression residuals.
    :param covar_style: How to compute the covariance matrix. Either 'sample' or 'hac'.
    :param kernel:
        The kernel function or name.
        If it's a string, it should match one of the `arch` package variance estimator.
        If it's a callable, it should return a covariance matrix.
    :param bw: Bandwidth for the kernel. If None, the kernel might compute the optimal bandwidth.
    :param kernel_kwargs: Additional keyword arguments to be passed to the kernel function.
    :return np.array: The computed covariance matrix omega.
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    if covar_style == "sample":
        return (reg - Dbar).T @ (reg - Dbar) / (len(reg) - 1)
    elif covar_style == "hac":
        if callable(kernel):
            return kernel(reg, **kernel_kwargs)
        elif isinstance(kernel, str) and hasattr(kernels, kernel):  # Arch covariance
            kerfunc = getattr(kernels, kernel)
            ker = kerfunc(reg, bandwidth=bw, **kernel_kwargs)
            return ker.cov.long_run
        else:
            raise NotImplementedError("Kernel not recognized or not implemented")
    else:
        raise ValueError(f"Unsupported covariance style: {covar_style}")



def Diebold_Mariano_pvalue(s1, s2):
    d_lst = s1.values - s2.values
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,1):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/float(len(d_lst))
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((float(len(d_lst))+1-2*1+1*(1-1)/float(len(d_lst)))/float(len(d_lst)))**(0.5)
    DM_stat = harvey_adj*DM_stat

    #return 2*scipy.stats.t.cdf(-abs(DM_stat), df = float(len(d_lst)) - 1)
    return round(scipy.stats.t.cdf(DM_stat, df = float(len(d_lst)) - 1),4)

def plot_Diebold_Mariano_pvalues(forecasts):


    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns)

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:

                # Find p-value
                p_values.loc[model1, model2] = Diebold_Mariano_pvalue(forecasts.loc[:, model1], forecasts.loc[:, model2])


                #w_stat, p_value = stats.wilcoxon(forecasts.loc[:, model1].values - forecasts.loc[:, model2].values, alternative='less')
                #p_values.loc[model1, model2] = p_value

    p_values['# models outprf'] = (p_values < 0.05).sum(axis=1)
    p_values.iloc[:,0:-1] = p_values.iloc[:,0:-1].astype(float)[p_values.iloc[:,0:-1]<0.05]


    fig, ax = plt.subplots(1, 1, figsize =(4,6))
    g = sns.heatmap(p_values.astype(float), linewidths=0.5, linecolor='black', square=True, cmap=sns.color_palette("mako", n_colors=1),cbar=False)

    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.1, top - 0.5)

    left, right = g.get_xlim()
    g.set_xlim(left, right - 0.98)


    for i in range(p_values.shape[0]):
        ax.text(17.5, i + 0.5, p_values.iloc[i, -1], ha='center', va='center', color='black', fontweight='bold', fontsize=9)

    ax.text(18.5, 16.5, 'Number of models outperformed by the corresponding model on the Y-axis', rotation=270, ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)


    plt.show()

def RPS_calculation(truth, results):

    #Evaluate submission
    rps_sub = []
    frc = np.cumsum(results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, axis=1)
    target = np.cumsum(truth.values, axis=1)

    return(np.round(np.mean(np.mean(np.power((target - frc), 2), axis=1)), 4))

def calc_pauc(y_test, y_hat, fpr=0.2):
    pAUC = 0
    for i in range(5):
        pAUC += roc_auc_score((y_test[:,i] > 0) * 1, y_hat[:,i], max_fpr=fpr)

    return pAUC / 5

def calc_ece(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def calculate_tables(universe, file_name, data_file, back, freq):

    m_names = ['KDE','GM','MND','GC','NF','DeepAR','GAN','VAE','Lag-Llama','NB','SR','LGBM','RF','SVM','MLP','BVAR','PatchTST']#,'EWMA']

    # Read Data
    #######################################
    cats = pd.read_excel('Categorizations.xlsx', sheet_name='Models').rename(columns={'Unnamed: 0':'Models'})
    asset_cats = pd.read_excel('Categorizations.xlsx', sheet_name='Assets').rename(columns={'Unnamed: 0':'Assets'})
    information_cal = pd.read_excel(file_name, sheet_name='Infocalib').set_index('Date')
    information = pd.read_excel(file_name, sheet_name='Info').set_index('Date')
    rps_assets = pd.read_excel(file_name, sheet_name='RPS_assets').rename(columns={'Unnamed: 0':'Assets'}).set_index('Assets')
    forecasts_data = pd.read_excel(file_name, sheet_name='forecasts')
    mappings = pd.read_excel(data_file,sheet_name='Tickers')
    closing_prices_df = pd.read_excel(data_file,sheet_name='Close')
    ff = pd.read_excel('F-F_Research_Data_Factors_daily.xlsx').rename(columns={'Unnamed: 0':'Date'}).set_index('Date')
    #######################################

    tickers = forecasts_data.iloc[2:102,1].values.tolist()
    forecasts = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(d, asset) for d in forecasts_data.iloc[2:,0][~pd.isna(forecasts_data.iloc[2:,0])].values for asset in tickers], names=['Date', 'Asset']),
                columns=pd.MultiIndex.from_tuples([(m, r) for m in [name for name in forecasts_data.columns if "Unnamed" not in name][1:len(m_names)+1]+['truth'] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
                dtype=float
        )

    forecasts.iloc[:,:] = pd.concat([forecasts_data.iloc[2:,2:len(m_names) * 5 + 2], forecasts_data.iloc[2:,-5:]], axis=1).values


    mappings = mappings[['Ticker Factset', 'Ticker']].set_index('Ticker Factset').to_dict()['Ticker']
    mappings['Unnamed: 0'] = 'Date'

    closing_prices_df = closing_prices_df.rename(columns=mappings).set_index('Date')
    closing_prices_df.index = pd.to_datetime(closing_prices_df.index, format='%d/%m/%Y')
    closing_prices_df = closing_prices_df.resample('B').last()
    closing_prices_df = closing_prices_df.iloc[1:,:]
    closing_prices_df = closing_prices_df.ffill(axis=0)
    rets = closing_prices_df.pct_change()
    rets = rets[(rets.index >= forecasts.index.get_level_values(0).unique()[0]) & (rets.index <= forecasts.index.get_level_values(0).unique()[-1])]
    rets = rets.merge(ff[['Mkt-RF','RF']] / 100,right_index=True,left_index=True)


    id_vol = {}
    for i in closing_prices_df.columns:
        nonna_sample = ~rets.loc[:,i].isna()
        y = rets.loc[nonna_sample,i].values.reshape(-1, 1) - rets.loc[nonna_sample,'RF'].values.reshape(-1, 1)
        x = rets.loc[nonna_sample,'Mkt-RF'].values.reshape(-1, 1)
        reg = LinearRegression().fit(y, x)
        id_vol[i] = np.sqrt(250) * np.std(y - reg.predict(x))

    results = pd.DataFrame(dict(sorted(id_vol.items(), key=lambda item: item[1])).items())
    vol = {}
    vol['Low'] = results.loc[results[1] <= results[1].describe([0.33, 0.67])['33%'], 0].values
    vol['Medium'] = results.loc[(results[1] > results[1].describe([0.33, 0.67])['33%']) & (results[1] <= results[1].describe([0.33, 0.67])['67%']), 0].values
    vol['High'] = results.loc[results[1] > results[1].describe([0.33, 0.67])['67%'], 0].values


    print('\n--------------Table 9---------------------')
    print('p-val difference in RPS between calibrated and uncalibrated models:')
    for c in m_names:
        print(c, round(information[c+'_RPS'].iloc[back:].mean(),4), round(information_cal[c+'_RPS'].iloc[back:].mean(),4), Diebold_Mariano_pvalue(information_cal[c+'_RPS'].iloc[back:], information[c+'_RPS'].iloc[back:]))

    keep_colummns = [item for item in information.columns if any(element in item for element in m_names)]
    information = information[keep_colummns]

    rps = information[[c for c in keep_colummns if '_RPS' in c]]
    rps.columns = m_names

    acc = information[[c for c in keep_colummns if '_Acc' in c]]
    kl = information[[c for c in keep_colummns if '_KL' in c]]
    ece = information[[c for c in keep_colummns if '_ECE' in c]]

    acc.columns = m_names
    kl.columns = m_names
    ece.columns = m_names


    print('\n\n--------------Table 8--------------------')
    print('RPS by ETF asset class')
    rps_by_sector = pd.DataFrame(
                index=m_names,
                columns=asset_cats['Sector'].unique(),
                data=0,
                dtype=float
            )
    for c in asset_cats['Sector'].unique():
        rps_by_sector.loc[:, c] = rps_assets.loc[rps_assets.index.isin(rps_assets.index[asset_cats['Sector'] == c]), :].mean()
    print(rps_by_sector.mean())

    print('\nRPS by Asset type')
    rps_by_type = pd.DataFrame(
                index=m_names,
                columns=asset_cats['Type'].unique(),
                data=0,
                dtype=float
            )
    for c in asset_cats['Type'].unique():
        rps_by_type.loc[:, c] = rps_assets.loc[rps_assets.index.isin(rps_assets.index[asset_cats['Type'] == c]), :].mean()
    print(rps_by_type.mean())


    print('\nRPS by Idiosyncratic vol')
    rps_by_vol = pd.DataFrame(
                index=m_names,
                columns=['Low','Medium','High'],
                data=0,
                dtype=float
            )
    for c in ['Low','Medium','High']:
        rps_by_vol.loc[:, c] = rps_assets.loc[rps_assets.index.isin(vol[c]), :].mean()
    print(rps_by_vol.mean())

    # Asset type for asset with highest RPS
    #print(asset_cats.loc[asset_cats['Assets'].isin(rps_assets.mean(axis=1).sort_values().tail(20).index),'Type'])

    if universe == 'M6+':
        # Relationship among RPS, ACC and ECE
        import statsmodels.api as sm
        X = sm.add_constant(np.concatenate([information[[c for c in information.columns if '_Acc' in c]].iloc[back:].mean().values.reshape(-1,1), information[[c for c in information.columns if '_ECE' in c]].iloc[back:].mean().values.reshape(-1,1)], axis=1))
        model = sm.OLS(information[[c for c in information.columns if '_RPS' in c]].iloc[back:].mean().values, X).fit()
        print('\n\n--------------Comment 4, section 3.1--------------------')
        print(model.summary())



    print('\n\n-------------Table 2 or 3--------------------')

    print('\nRPS')
    temp = pd.concat([information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].mean(), information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].mean().rank()], axis=1)
    print(temp.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1))

    print('\nUpper 2.5% RPS')
    print(round(information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].mean() + t.ppf(1 - (1 - 0.95) / 2, information.iloc[back:].shape[0] - 1) * information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].std() / np.sqrt(information.iloc[back:].shape[0]), 5))

    print('\nW_RPS')
    temp = pd.concat([information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].describe([0.05,0.95]).loc['95%',:], information[[c for c in keep_colummns if '_RPS' in c]].iloc[back:].describe([0.05,0.95]).loc['95%',:].rank()], axis=1)
    print(temp.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1))

    print('\npAUC')
    temp = pd.concat([information[[c for c in keep_colummns if '_pAUC' in c]].iloc[back:].mean(), information[[c for c in keep_colummns if '_pAUC' in c]].shape[1] + 1 - information[[c for c in keep_colummns if '_pAUC' in c]].iloc[back:].mean().rank()], axis=1)
    print(temp.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1))

    print('\npAUC Bootstrap upper 2.5%')
    print(round(information[[c for c in keep_colummns if '_PAUCBoot' in c]].iloc[back:].mean(), 4))

    print('\nExpected Calibration Error')
    temp = pd.concat([100 * information[[c for c in keep_colummns if '_ECE' in c]].iloc[back:].mean(), information[[c for c in keep_colummns if '_ECE' in c]].iloc[back:].mean().rank()], axis=1)
    print(temp.apply(lambda row: f"{row[0]:.2f} ({row[1]:.0f})", axis=1))

    print('\nAccuracies')
    temp = pd.concat([100 * information[[c for c in keep_colummns if '_Acc' in c]].iloc[back:].mean(), information[[c for c in keep_colummns if '_Acc' in c]].shape[1] + 1 - information[[c for c in keep_colummns if '_Acc' in c]].iloc[back:].mean().rank()], axis=1)
    print(temp.apply(lambda row: f"{row[0]:.1f} ({row[1]:.0f})", axis=1))

    print('\nACC Bootstrap lower 2.5%')
    print(round(100 * information[[c for c in keep_colummns if '_ACCBoot' in c]].iloc[back:].mean(), 4))


    if universe == 'M6+':
        print('\n\n--------------Table 5--------------------')
        information.columns = pd.MultiIndex.from_tuples(
            [col.split('_') for col in keep_colummns],
            names=['model', 'metric'])
        information = information.iloc[back:,:]

        df_long = information.stack(level=0).reset_index().drop(columns=['KL','ECE','PAUCBoot','ACCBoot'])
        def min_max_normalize(column):
            return (column - column.min()) / (column.max() - column.min())

        # Apply normalization to the 'model_metric' column
        for c in ['Acc','RPS','pAUC']:
            df_long[c] = min_max_normalize(df_long[c])
        df_long['pAUC'] = 1 - df_long['pAUC']
        df_long['Acc'] = 1 - df_long['Acc']

        df_long = df_long.melt(id_vars=['Date','model'], value_vars = ['Acc','RPS','pAUC'])

        a = pd.DataFrame(index=df_long['model'].unique(), columns=range(0,1000))
        for i in range(0,1000):
            #d = np.random.choice(range(0,df_long['Date'].nunique() - 12), size=1)[0]
            #bootstrap_problems = df_long['Date'].unique()[d:d+12]
            bootstrap_problems = np.random.choice(df_long['Date'].unique(), size=12, replace=True)
            bootstrap_metrics = np.random.choice(df_long['metric'].unique(), size=3, replace=True)
            df_sampled = df_long[df_long['Date'].isin(bootstrap_problems) & df_long['metric'].isin(bootstrap_metrics)]

            a.loc[:,i] = df_sampled.groupby('model')['value'].mean().rank().values

        b = pd.DataFrame(index=df_long['model'].unique(), columns=range(1,18))
        for i in range(1,18):
            b.loc[:,i] = a[a==i].count(axis=1) / 10

        print(b.loc[m_names])

    print('\n\n-------------Figure 2--------------------')
    plot_Diebold_Mariano_pvalues(rps.iloc[back:,:])



    mcs = MCS(rps.iloc[back:,:], size=0.05, seed=42)
    mcs.compute()
    print('\n\n-------------Table 4--------------------')
    print(mcs.pvalues > 0.05)



    print('\n\n-------------Table 7--------------------')
    print('\nModel Category 1')
    avgs = {}
    for c in cats['Cat1'].unique():
        avgs[c] = rps.iloc[back:,:][cats.loc[cats['Cat1'] == c, 'Models'].values].mean(axis=1)

    for c1 in cats['Cat1'].unique():
        print(c1, round(avgs[c1].mean(),4))
        for c2 in cats['Cat1'].unique():
            if c1 != c2:
                print(c1, 'vs.', c2, Diebold_Mariano_pvalue(avgs[c1], avgs[c2]))

    print('\nModel Category 2')
    avgs = {}
    for c in cats['Cat2'].unique():
        avgs[c] = rps.iloc[back:,:][cats.loc[cats['Cat2'] == c, 'Models'].values].mean(axis=1)

    for c1 in cats['Cat2'].unique():
        print(c1, round(avgs[c1].mean(),4))
        for c2 in cats['Cat2'].unique():
            if c1 != c2:
                print(c1, 'vs.', c2, Diebold_Mariano_pvalue(avgs[c1], avgs[c2]))

    print('\nModel Complexity')
    avgs = {}
    avgs['Low'] = rps.iloc[back:,:][cats.loc[cats['Complexity'] < cats['Complexity'].describe([0.33, 0.67])['33%'], 'Models'].values].mean(axis=1)
    avgs['Medium'] = rps.iloc[back:,:][cats.loc[(cats['Complexity'] >= cats['Complexity'].describe([0.33, 0.67])['33%']) & (cats['Complexity'] <= cats['Complexity'].describe([0.33, 0.67])['67%']), 'Models'].values].mean(axis=1)
    avgs['High'] = rps.iloc[back:,:][cats.loc[cats['Complexity'] > cats['Complexity'].describe([0.33, 0.67])['67%'], 'Models'].values].mean(axis=1)


    for c1 in avgs.keys():
        print(c1, round(avgs[c1].mean(),4))
        for c2 in avgs.keys():
            if c1 != c2:
                print(c1, 'vs.', c2, Diebold_Mariano_pvalue(avgs[c1], avgs[c2]))


    print('\nModel Smoothness')
    avgs = {}
    avgs['Low'] = rps.iloc[back:,:][cats.loc[cats['Smoothness'] < cats['Smoothness'].describe([0.33, 0.67])['33%'], 'Models'].values].mean(axis=1)
    avgs['Medium'] = rps.iloc[back:,:][cats.loc[(cats['Smoothness'] >= cats['Smoothness'].describe([0.33, 0.67])['33%']) & (cats['Smoothness'] <= cats['Complexity'].describe([0.33, 0.67])['67%']), 'Models'].values].mean(axis=1)
    avgs['High'] = rps.iloc[back:,:][cats.loc[cats['Smoothness'] > cats['Smoothness'].describe([0.33, 0.67])['67%'], 'Models'].values].mean(axis=1)


    for c1 in avgs.keys():
        print(c1, round(avgs[c1].mean(),4))
        for c2 in avgs.keys():
            if c1 != c2:
                print(c1, 'vs.', c2, Diebold_Mariano_pvalue(avgs[c1], avgs[c2]))


    print('\n\n\nRun ensemble techniques.....')
    score_rps_top = []
    score_kl_acc2_top = []
    score_unc = []

    pauc_rps_top = []
    pauc_kl_acc2_top = []
    pauc_unc = []

    ece_rps_top = []
    ece_kl_acc2_top = []
    ece_unc = []

    acc_rps_top = []
    acc_kl_acc2_top = []
    acc_unc = []

    TOP_N = 4



    for i in range(back, rps.shape[0]):
        rps_data = rps.iloc[i - back:i,:]

        ece_data = ece.iloc[i - back:i,:]

        avgs = rps_data.mean()
        rps_score = (-avgs).rank()
        kl_data = kl.iloc[i - back:i,:]
        avgs_kl = kl_data.mean()
        kl_acc_score2 = ((avgs_kl > avgs_kl.describe([0.2,0.8])['20%']) & (avgs_kl < avgs_kl.describe([0.2,0.8])['80%'])) * (-avgs).rank()
        d = rps.iloc[i,:].name


        top_m_names = pd.Index(m_names)

        truth = pd.DataFrame(columns=["Rank1", "Rank2", "Rank3", "Rank4","Rank5"], data = forecasts.loc[d, ('truth', )].values)

        ################################################################################
        if (i < rps.shape[0] - 1) & ((i % freq == freq - 1) | (i == back)):
            print('Rebal at date: ', d)
            # Perform the unconditional mcs with an HAC estimator
            mcs, S, cval, pval, removed = cmcs(rps_data[top_m_names].values, covar_style="hac", kernel="Bartlett")
            #print('Unconditional models')
            unc_models = top_m_names[mcs.reshape(-1,) == 1].values
            #print(unc_models)


        results = pd.DataFrame(columns=["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results['ID'] = tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = 0
        for c in unc_models:
            results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += forecasts.loc[d, (c, )].values

        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] /= len(unc_models)


        score_unc.append(RPS_calculation(truth, results))
        pauc_unc.append(calc_pauc(truth.values, results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, fpr=0.25))
        ece_unc.append(calc_ece(results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, truth.values.argmax(axis=1)))
        acc_unc.append(sum(truth.values.argmax(axis=1) == results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values.argmax(axis=1)))



        ################################################################################
        if (i % freq == freq - 1) | (i == back):
            top_4 = [c for c, v in zip(rps_score.index, rps_score) if v >= heapq.nlargest(TOP_N, rps_score.values)[-1]]


        results = pd.DataFrame(columns=["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results['ID'] = tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = 0
        for c in top_4:
            results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += forecasts.loc[d, (c, )].values

        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] /= len(top_4)


        score_rps_top.append(RPS_calculation(truth, results))
        pauc_rps_top.append(calc_pauc(truth.values, results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, fpr=0.25))
        ece_rps_top.append(calc_ece(results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, truth.values.argmax(axis=1)))
        acc_rps_top.append(sum(truth.values.argmax(axis=1) == results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values.argmax(axis=1)))

        ################################################################################
        if (i % freq == freq - 1) | (i == back):
            top_kl_acc = [c for c, v in zip(kl_acc_score2.index, kl_acc_score2) if v >= heapq.nlargest(TOP_N, kl_acc_score2.values)[-1]]


        results = pd.DataFrame(columns=["ID", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        results['ID'] = tickers
        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] = 0
        for c in top_kl_acc:
            results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] += forecasts.loc[d, (c, )].values

        results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]] /= len(top_kl_acc)


        score_kl_acc2_top.append(RPS_calculation(truth, results))
        pauc_kl_acc2_top.append(calc_pauc(truth.values, results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, fpr=0.25))
        ece_kl_acc2_top.append(calc_ece(results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, truth.values.argmax(axis=1)))
        acc_kl_acc2_top.append(sum(truth.values.argmax(axis=1) == results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values.argmax(axis=1)))


    print('\n\n-------------Table 10--------------------')
    results_top = pd.DataFrame(columns=['ENS_MCS','ENS_SMOOTH_RPS','ENS_RPS'], index=['RPS','W_RPS','pAUC', 'ECE','ACC'])
    results_top['ENS_MCS'] = [np.round(np.mean(score_unc), 5), np.round(np.percentile(score_unc, 95), 4), np.round(np.mean(pauc_unc), 4), np.round(np.mean(ece_unc), 4), np.round(np.mean(acc_unc), 1)]
    results_top['ENS_SMOOTH_RPS'] = [np.round(np.mean(score_kl_acc2_top), 5), np.round(np.percentile(score_kl_acc2_top, 95), 4), np.round(np.mean(pauc_kl_acc2_top), 4), np.round(np.mean(ece_kl_acc2_top), 4), np.round(np.mean(acc_kl_acc2_top), 1)]
    results_top['ENS_RPS'] = [np.round(np.mean(score_rps_top), 5), np.round(np.percentile(score_rps_top, 95), 4), np.round(np.mean(pauc_rps_top), 4), np.round(np.mean(ece_rps_top), 4), np.round(np.mean(acc_rps_top), 1)]
    print(results_top.T.sort_values(by=['RPS','W_RPS']))




if __name__ == '__main__':

    # To run: python Calculate_tables.py --file_name Results_m6.xlsx --tuning_sample 12 --M6 1

    parser = argparse.ArgumentParser(description='Calculate metrics')
    parser.add_argument('--file_name', nargs='+', type=str, help="The file that contains the forecasts")
    parser.add_argument('--tuning_sample', nargs='?', type=int, const=0, default=12)
    parser.add_argument('--freq', nargs='?', type=int, const=0, default=6)
    parser.add_argument('--M6', nargs='?', type=int, const=1, default=1)
    args = parser.parse_args()

    if args.M6:
        universe = 'M6'
        data_file = 'Data_M6.xlsx'
    else:
        universe = 'M6+'
        data_file = 'Data_v2.xlsx'

    calculate_tables(universe, args.file_name, data_file, args.tuning_sample, args.freq)


    print('\nTask completed...')
    print('\n###################################################################################')