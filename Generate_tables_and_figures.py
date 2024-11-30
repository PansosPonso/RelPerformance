import statsmodels.api as sm
import scipy
from arch.bootstrap import MCS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import heapq
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import t, sem
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#################### FUNCTIONS ####################

# This function returns the pvalue from comparing the RPS series s1 and s2, using the Diebold Mariano test
def Diebold_Mariano_pvalue(s1, s2):

    d_lst = s1.values - s2.values
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / T) * autoCov
    gamma = []
    for lag in range(0,1):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / float(len(d_lst))
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((float(len(d_lst)) - 1) / float(len(d_lst))) ** (0.5)
    DM_stat = harvey_adj * DM_stat

    return round(scipy.stats.t.cdf(DM_stat, df = float(len(d_lst)) - 1), 4)

# Function that produces Figure 2
def plot_Diebold_Mariano_pvalues(forecasts, universe):

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


    p_values['# models outprf'] = (p_values < 0.05).sum(axis=1)
    p_values.iloc[:,0:-1] = p_values.iloc[:,0:-1].astype(float)[p_values.iloc[:,0:-1]<0.05]


    fig, ax = plt.subplots(1, 1, figsize =(8,10))
    g = sns.heatmap(p_values.astype(float), linewidths=0.5, linecolor='black', square=True, cmap=sns.color_palette("mako", n_colors=1),cbar=False)

    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.1, top - 0.5)

    left, right = g.get_xlim()
    g.set_xlim(left, right - 0.98)


    for i in range(p_values.shape[0]):
        ax.text(17.5, i + 0.5, p_values.iloc[i, -1], ha='center', va='center', color='black', fontweight='bold', fontsize=9)

    ax.text(18.5, 16.5, 'Number of models outperformed by the corresponding model on the Y-axis', rotation=270, ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)

    if universe == 'M6+':
        title = 'results/Figure 2 (M6+).jpg'
    elif universe == 'M6':
        title = 'results/Figure 2 (M6).jpg'
    else:
        title = 'results/Figure 2.jpg'

    plt.tight_layout(pad=0.5, rect=[0, 0.01, 1, 0.99])
    plt.savefig(title)

# Function that calculates the RPS
def RPS_calculation(truth, results):

    #Evaluate submission
    rps_sub = []
    frc = np.cumsum(results[["Rank1", "Rank2", "Rank3", "Rank4","Rank5"]].values, axis=1)
    target = np.cumsum(truth.values, axis=1)

    return(np.round(np.mean(np.mean(np.power((target - frc), 2), axis=1)), 4))

# Function that calculates the pAUC
def calc_pauc(y_test, y_hat, fpr=0.2):
    pAUC = 0
    for i in range(5):
        pAUC += roc_auc_score((y_test[:,i] > 0) * 1, y_hat[:,i], max_fpr=fpr)

    return pAUC / 5

# Function that calculates the ECE
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

# Function that generates Figure 1
def generate_figure1():
    information = pd.read_excel('outputs/Results_M6_with_metrics.xlsx', sheet_name='Info').set_index('Date')
    rps_M6 = information[[c for c in information.columns if '_RPS' in c]]
    rps_M6.columns = rps_M6.columns.str.replace('_RPS$', '', regex=True)
    rps_M6 = rps_M6.iloc[12:,:]

    information = pd.read_excel('outputs/Results_v2_with_metrics.xlsx', sheet_name='Info').set_index('Date')
    rps_v2 = information[[c for c in information.columns if '_RPS' in c]]
    rps_v2.columns = rps_v2.columns.str.replace('_RPS$', '', regex=True)
    rps_v2 = rps_v2.iloc[36:,:]

    
    rps_M6_mean = rps_M6.mean()
    rps_M6_ci = sem(rps_M6, axis=0) * t.ppf((1 + 0.95) / 2, rps_M6.shape[0] - 1)

    rps_v2_mean = rps_v2.mean()
    rps_v2_ci = sem(rps_v2, axis=0) * t.ppf((1 + 0.95) / 2, rps_v2.shape[0] - 1)

    rps_M6_ci = pd.Series(rps_M6_ci, index=rps_M6.columns)
    rps_v2_ci = pd.Series(rps_v2_ci, index=rps_v2.columns)

    rps_M6_mean["Naive"] = 0.16
    rps_v2_mean["Naive"] = 0.16

    rps_M6_ci["Naive"] = 0
    rps_v2_ci["Naive"] = 0


    sorted_indices = rps_M6_mean.sort_values().index
    rps_M6_mean = rps_M6_mean[sorted_indices]
    rps_M6_ci = rps_M6_ci[sorted_indices]
    rps_v2_mean = rps_v2_mean[sorted_indices]
    rps_v2_ci = rps_v2_ci[sorted_indices]

    bar_width = 0.35
    x = np.arange(len(rps_M6.columns)+1)  # positions
    x2 = x + bar_width

    fig, ax = plt.subplots(figsize=(15, 8))

    bars1 = ax.bar(x, rps_M6_mean, bar_width, label='M6 sample', capsize=5, color='lightcoral', alpha=0.7)
    bars2 = ax.bar(x2, rps_v2_mean, bar_width, label='M6+ sample', capsize=5, color='skyblue', alpha=0.7)

    ax.errorbar(x, rps_M6_mean, yerr=rps_M6_ci, fmt='none', ecolor='black', elinewidth=1, capsize=2, alpha=0.4)
    ax.errorbar(x2, rps_v2_mean, yerr=rps_v2_ci, fmt='none', ecolor='black', elinewidth=1, capsize=2, alpha=0.4)

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlabel('Models')
    ax.set_ylabel('RPS')
    ax.set_xticks(x + bar_width / 2)  # Center the ticks between the groups
    ax.set_xticklabels(rps_M6_mean.index)
    ax.set_ylim(0.15, 0.175)
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/Figure 1.jpg')


# The main function that produces all Tables and Figures in the paper
def calculate_tables(universe, file_name, back, freq):

    print(f'\n\nGenerating results for {universe}')

    # Read Data
    # file_name contains the forecasts and all metrics necessary for the Tables
    # 'Categorizations.xlsx' contains all categorizations referenced in Table 1
    # 'F-Idiosyncratic_vol.xlsx' contains data for the idiosyncratic volatility of each asset
    # 'summary_leaderboard.xlsx' contains data for the RPS scores of the competitors in the M6 competition
    #######################################
    cats = pd.read_excel('data/Categorizations.xlsx', sheet_name='Models').rename(columns={'Unnamed: 0':'Models'})
    asset_cats = pd.read_excel('data/Categorizations.xlsx', sheet_name='Assets').rename(columns={'Unnamed: 0':'Assets'})
    information_cal = pd.read_excel(file_name, sheet_name='Infocalib').set_index('Date')
    information = pd.read_excel(file_name, sheet_name='Info').set_index('Date')
    rps_assets = pd.read_excel(file_name, sheet_name='RPS_assets').rename(columns={'Unnamed: 0':'Assets'}).set_index('Assets')
    forecasts_data = pd.read_excel(file_name, sheet_name='forecasts')
    id_vol = pd.read_excel('outputs/Idiosyncratic_vol.xlsx',sheet_name=universe)
    competitors = pd.read_excel('data/summary_leaderboard.xlsx',sheet_name='RPS')
    #######################################

    m_names = [name for name in list(forecasts_data.columns) if "Unnamed" not in name][1:-1]

    tickers = forecasts_data.iloc[2:102,1].values.tolist()
    forecasts = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(d, asset) for d in forecasts_data.iloc[2:,0][~pd.isna(forecasts_data.iloc[2:,0])].values for asset in tickers], names=['Date', 'Asset']),
                columns=pd.MultiIndex.from_tuples([(m, r) for m in [name for name in forecasts_data.columns if "Unnamed" not in name][1:] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
                dtype=float
        )
    forecasts.iloc[:,:] = pd.concat([forecasts_data.iloc[2:,2:len(m_names) * 5 + 2], forecasts_data.iloc[2:,-5:]], axis=1).values


    # vol contains the tickers for each idiosyncratic volatility tertile
    vol = {}
    vol['Low'] = id_vol.loc[id_vol[1] <= id_vol[1].describe([0.33, 0.67])['33%'], 0].values
    vol['Medium'] = id_vol.loc[(id_vol[1] > id_vol[1].describe([0.33, 0.67])['33%']) & (id_vol[1] <= id_vol[1].describe([0.33, 0.67])['67%']), 0].values
    vol['High'] = id_vol.loc[id_vol[1] > id_vol[1].describe([0.33, 0.67])['67%'], 0].values

    keep_colummns = [item for item in information.columns if any(element in item for element in m_names)]
    information = information[keep_colummns]

    rps = information[[c for c in keep_colummns if '_RPS' in c]]
    rps_cal = information_cal[[c for c in keep_colummns if '_RPS' in c]]
    acc = information[[c for c in keep_colummns if '_Acc' in c]]
    kl = information[[c for c in keep_colummns if '_KL' in c]]
    ece = information[[c for c in keep_colummns if '_ECE' in c]]
    pauc = information[[c for c in keep_colummns if '_pAUC' in c]]

    rps.columns = rps.columns.str.replace('_RPS$', '', regex=True)
    acc.columns = acc.columns.str.replace('_Acc$', '', regex=True)
    kl.columns = kl.columns.str.replace('_KL$', '', regex=True)
    ece.columns = ece.columns.str.replace('_ECE$', '', regex=True)
    pauc.columns = pauc.columns.str.replace('_pAUC$', '', regex=True)

    m_names = rps.columns.tolist()

    #--------------Table 2 or 3--------------------
    if universe == 'M6':
        title = 'results/Table 2'
    else:
        title = 'results/Table 3'

    # Caclulate average RPS, pAUC, ECE and ACC scores and W_RPS, together with their statistical significance
    temp_rps = pd.concat([rps.iloc[back:].mean(), rps.iloc[back:].mean().rank()], axis=1)
    temp_w_rps = pd.concat([rps.iloc[back:].describe([0.05,0.95]).loc['95%',:], rps.iloc[back:].describe([0.05,0.95]).loc['95%',:].rank()], axis=1)
    temp_pauc = pd.concat([pauc.iloc[back:].mean(), pauc.shape[1] + 1 - pauc.iloc[back:].mean().rank()], axis=1)
    temp_ece = pd.concat([100 * ece.iloc[back:].mean(), ece.iloc[back:].mean().rank()], axis=1)
    temp_acc = pd.concat([100 * acc.iloc[back:].mean(), acc.shape[1] + 1 - acc.iloc[back:].mean().rank()], axis=1)

    pd.DataFrame(data = np.stack([np.where((rps.iloc[back:].mean() + t.ppf(1 - (1 - 0.95) / 2, rps.iloc[back:].shape[0] - 1) * rps.iloc[back:].std() / np.sqrt(rps.iloc[back:].shape[0]) < 0.16).values, temp_rps.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values + '*', temp_rps.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values),
        np.where((rps.iloc[back:].describe([0.05,0.95]).loc['95%',:] < 0.16).values, temp_w_rps.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values + '*', temp_w_rps.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values),
        np.where((information[[c for c in keep_colummns if '_PAUCBoot' in c]].iloc[back:].mean() > 0.5).values, temp_pauc.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values + '*', temp_pauc.apply(lambda row: f"{row[0]:.4f} ({row[1]:.0f})", axis=1).values),
        np.where((ece.iloc[back:].mean() < 0.05).values, temp_ece.apply(lambda row: f"{row[0]:.2f} ({row[1]:.0f})", axis=1).values + '*', temp_ece.apply(lambda row: f"{row[0]:.2f} ({row[1]:.0f})", axis=1).values),
        np.where((information[[c for c in keep_colummns if '_ACCBoot' in c]].iloc[back:].mean() > 0.2).values, temp_acc.apply(lambda row: f"{row[0]:.1f} ({row[1]:.0f})", axis=1).values + '*', temp_acc.apply(lambda row: f"{row[0]:.1f} ({row[1]:.0f})", axis=1).values)]).T, index = m_names, columns = ['RPS','W_RPS','pAUC','ECE','ACC']).to_html(title+'.html')



    #--------------Table 9---------------------
    dm_pvals = np.zeros(len(m_names))
    for i,c in enumerate(m_names):
        dm_pvals[i] = Diebold_Mariano_pvalue(information_cal[c+'_RPS'].iloc[back:], information[c+'_RPS'].iloc[back:])

    if universe == 'M6+':
        title = 'results/Table 9 (Panel B).html'
    elif universe == 'M6':
        title = 'results/Table 9 (Panel A).html'
    else:
        title = 'results/Table 9.html'


    pd.DataFrame(data = np.stack([np.where((rps.iloc[back:].mean() + t.ppf(1 - (1 - 0.95) / 2, rps.iloc[back:].shape[0] - 1) * rps.iloc[back:].std() / np.sqrt(rps.iloc[back:].shape[0]) < 0.16).values, rps.iloc[back:].mean().apply(lambda row: f"{row:.4f}*").values, rps.iloc[back:].mean().apply(lambda row: f"{row:.4f}").values), np.where((rps_cal.iloc[back:].mean() + t.ppf(1 - (1 - 0.95) / 2, rps_cal.iloc[back:].shape[0] - 1) * rps_cal.iloc[back:].std() / np.sqrt(rps_cal.iloc[back:].shape[0]) < 0.16).values, rps_cal.iloc[back:].mean().apply(lambda row: f"{row:.4f}*").values, rps_cal.iloc[back:].mean().apply(lambda row: f"{row:.4f}").values), dm_pvals]).T, index = m_names, columns = ['Raw output','Calibrated output','p-value']).to_html(title)


    if 'EWMA' in rps.columns:
        rps = rps.drop(columns=['EWMA'])
        ece = ece.drop(columns=['EWMA'])
        kl = kl.drop(columns=['EWMA'])
        acc = acc.drop(columns=['EWMA'])

    m_names_ex_ewma = rps.columns.tolist()
    cats = cats.loc[cats['Models'].isin(m_names_ex_ewma), :].reset_index(drop=True)


    #--------------Table 6--------------------
    if universe == 'M6':
        comparisons = pd.DataFrame(index=m_names_ex_ewma, columns=['#','%'])
        
        # Keep scores from competitors that participated from the start
        comp = competitors.loc[~competitors['Global'].isna(), 'Global'] 
        for c in m_names_ex_ewma:
            comparisons.loc[c,'#'] = comp[comp < rps[c].iloc[back:].mean()].count()
            comparisons.loc[c,'%'] = round(100 * comp[comp < rps[c].iloc[back:].mean()].count() / comp.shape[0], 1)

        comparisons.sort_values(by=['#']).to_html('results/Table 6.html')

    #--------------Table 8--------------------
    rps_by_sector = pd.DataFrame(
                index=m_names_ex_ewma,
                columns=asset_cats['Sector'].unique(),
                data=0,
                dtype=float
            )
    for c in asset_cats['Sector'].unique():
        rps_by_sector.loc[:, c] = rps_assets.loc[rps_assets.index.isin(rps_assets.index[asset_cats['Sector'] == c]), :].mean()

    rps_by_type = pd.DataFrame(
                index=m_names_ex_ewma,
                columns=asset_cats['Type'].unique(),
                data=0,
                dtype=float
            )
    for c in asset_cats['Type'].unique():
        rps_by_type.loc[:, c] = rps_assets.loc[rps_assets.index.isin(rps_assets.index[asset_cats['Type'] == c]), :].mean()

    rps_by_vol = pd.DataFrame(
                index=m_names_ex_ewma,
                columns=['Low','Medium','High'],
                data=0,
                dtype=float
            )
    for c in ['Low','Medium','High']:
        rps_by_vol.loc[:, c] = rps_assets.loc[rps_assets.index.isin(vol[c]), :].mean()

    if universe == 'M6+':
        title = 'results/Table 8 (M6+).html'
    elif universe == 'M6':
        title = 'results/Table 8 (M6).html'
    else:
        title = 'results/Table 8.html'

    with open(title, 'w') as _file:
        _file.write('RPS by ETF asset class\n' + pd.DataFrame(data = np.where((rps_by_sector.mean() + t.ppf(1 - (1 - 0.95) / 2, rps_by_sector.shape[0] - 1) * rps_by_sector.std() / np.sqrt(rps_by_sector.shape[0]) < 0.16).values, rps_by_sector.mean().apply(lambda row: f"{row:.4f}*").values, rps_by_sector.mean().apply(lambda row: f"{row:.4f}").values), index = asset_cats['Sector'].unique(), columns = ['RPS']).to_html() + "\nRPS by Asset type\n" + pd.DataFrame(data = np.where((rps_by_type.mean() + t.ppf(1 - (1 - 0.95) / 2, rps_by_type.shape[0] - 1) * rps_by_type.std() / np.sqrt(rps_by_type.shape[0]) < 0.16).values, rps_by_type.mean().apply(lambda row: f"{row:.4f}*").values, rps_by_type.mean().apply(lambda row: f"{row:.4f}").values), index = asset_cats['Type'].unique(), columns = ['RPS']).to_html() + "\nRPS by Idiosyncratic vol\n" + pd.DataFrame(data = np.where((rps_by_vol.mean() + t.ppf(1 - (1 - 0.95) / 2, rps_by_vol.shape[0] - 1) * rps_by_vol.std() / np.sqrt(rps_by_vol.shape[0]) < 0.16).values, rps_by_vol.mean().apply(lambda row: f"{row:.4f}*").values, rps_by_vol.mean().apply(lambda row: f"{row:.4f}").values), index = ['Low','Medium','High'], columns = ['RPS']).to_html())


    if universe != 'M6':
        print('\n\n--------------Footnote 11, section 3.4--------------------')
        num_etfs = (asset_cats.loc[asset_cats['Assets'].isin(rps_assets.mean(axis=1).sort_values().tail(20).index),'Type'] == 'ETF').sum()
        print(f'\nAmong the 20 assets with highest RPS, {num_etfs} are ETFs')
        num_etfs = (asset_cats.loc[asset_cats['Assets'].isin(rps_assets.mean(axis=1).sort_values().head(20).index),'Type'] == 'ETF').sum()
        print(f'\nAmong the 20 assets with lowest RPS, {num_etfs} are ETFs')

    if universe != 'M6':
        print('\n\n--------------Comment 4, section 3.1--------------------')
        # Relationship among RPS, ACC and ECE using simple OLS
        import statsmodels.api as sm
        X = sm.add_constant(np.concatenate([information[[c for c in information.columns if '_Acc' in c]].iloc[back:].mean().values.reshape(-1,1), information[[c for c in information.columns if '_ECE' in c]].iloc[back:].mean().values.reshape(-1,1)], axis=1))
        model = sm.OLS(information[[c for c in information.columns if '_RPS' in c]].iloc[back:].mean().values, X).fit()
        print(model.summary())


    if universe != 'M6':
        #--------------Table 5--------------------
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
            bootstrap_problems = np.random.choice(df_long['Date'].unique(), size=12, replace=True)
            bootstrap_metrics = np.random.choice(df_long['metric'].unique(), size=3, replace=True)
            df_sampled = df_long[df_long['Date'].isin(bootstrap_problems) & df_long['metric'].isin(bootstrap_metrics)]

            a.loc[:,i] = df_sampled.groupby('model')['value'].mean().rank().values

        b = pd.DataFrame(index=df_long['model'].unique(), columns=range(1,4))
        for i in range(1,4):
            b.loc[:,i] = a[a==i].count(axis=1) / 10

        b.loc[m_names_ex_ewma].to_html('results/Table 5.html')

    #-------------Figure 2--------------------
    plot_Diebold_Mariano_pvalues(rps.iloc[back:,:], universe)


    #-------------Table 4--------------------
    mcs = MCS(rps.iloc[back:,:], size=0.05, seed=RANDOM_SEED)
    mcs.compute()

    if universe == 'M6+':
        title = 'results/Table 4 (M6+).html'
    elif universe == 'M6':
        title = 'results/Table 4 (M6).html'
    else:
        title = 'results/Table 4.html'

    # Report the models that belongt to MCS at the 95% statistical level
    (mcs.pvalues.loc[m_names_ex_ewma] > 0.05).to_html(title)



    #------------Table 7--------------------
    if universe == 'M6+':
        title = 'results/Table 7 (M6+).html'
    elif universe == 'M6':
        title = 'results/Table 7 (M6).html'
    else:
        title = 'Table 7.html'
    
    #Model Category 1
    avgs = pd.DataFrame(columns=cats['Cat1'].unique())
    for c in cats['Cat1'].unique():
        avgs[c] = rps.iloc[back:,:][cats.loc[cats['Cat1'] == c, 'Models'].values].mean(axis=1)

    # Caclulate all average RPS scores per category and then the Diebold-Mariano p values for all pairwise comparisons
    avg_rps = pd.DataFrame(index=cats['Cat1'].unique())
    dm_pvals = {}
    for c1 in cats['Cat1'].unique():
        avg_rps[c1] = round(avgs[c1].mean(),4)
        for c2 in cats['Cat1'].unique():
            if c1 != c2:
                dm_pvals[c1 + ' vs. ' + c2] = Diebold_Mariano_pvalue(avgs[c1], avgs[c2])

    with open(title, 'w') as _file:
        _file.write('Model Category 1\n' + pd.DataFrame(data = np.where((avgs.mean() + t.ppf(1 - (1 - 0.95) / 2, avgs.shape[0] - 1) * avgs.std() / np.sqrt(avgs.shape[0]) < 0.16).values, avg_rps.mean().apply(lambda row: f"{row:.4f}*").values, avg_rps.mean().apply(lambda row: f"{row:.4f}").values), index = cats['Cat1'].unique(), columns = ['RPS']).to_html(index=False) + "\nDiebold-Mariano test\n" + pd.DataFrame.from_dict(dm_pvals.items()).rename(columns={0:' ',1:'p-values'}).to_html(index=False))

    #Model Category 2
    avgs = pd.DataFrame(columns=cats['Cat2'].unique())
    for c in cats['Cat2'].unique():
        avgs[c] = rps.iloc[back:,:][cats.loc[cats['Cat2'] == c, 'Models'].values].mean(axis=1)

    # Caclulate all average RPS scores per category and then the Diebold-Mariano p values for all pairwise comparisons
    avg_rps = pd.DataFrame(index=cats['Cat2'].unique())
    dm_pvals = {}
    for c1 in cats['Cat2'].unique():
        avg_rps[c1] = round(avgs[c1].mean(),4)
        for c2 in cats['Cat2'].unique():
            if c1 != c2:
                dm_pvals[c1 + ' vs. ' + c2] = Diebold_Mariano_pvalue(avgs[c1], avgs[c2])

    with open(title, 'a') as _file:
        _file.write('Model Category 2\n' + pd.DataFrame(data = np.where((avgs.mean() + t.ppf(1 - (1 - 0.95) / 2, avgs.shape[0] - 1) * avgs.std() / np.sqrt(avgs.shape[0]) < 0.16).values, avg_rps.mean().apply(lambda row: f"{row:.4f}*").values, avg_rps.mean().apply(lambda row: f"{row:.4f}").values), index = cats['Cat2'].unique(), columns = ['RPS']).to_html(index=False) + "\nDiebold-Mariano test\n" + pd.DataFrame.from_dict(dm_pvals.items()).rename(columns={0:' ',1:'p-values'}).to_html(index=False))


    #Model Complexity
    avgs = pd.DataFrame(columns=['Low','Medium','High'])
    avgs['Low'] = rps.iloc[back:,:][cats.loc[cats['Complexity'] < cats['Complexity'].describe([0.33, 0.67])['33%'], 'Models'].values].mean(axis=1)
    avgs['Medium'] = rps.iloc[back:,:][cats.loc[(cats['Complexity'] >= cats['Complexity'].describe([0.33, 0.67])['33%']) & (cats['Complexity'] <= cats['Complexity'].describe([0.33, 0.67])['67%']), 'Models'].values].mean(axis=1)
    avgs['High'] = rps.iloc[back:,:][cats.loc[cats['Complexity'] > cats['Complexity'].describe([0.33, 0.67])['67%'], 'Models'].values].mean(axis=1)
    
    avg_rps = pd.DataFrame(index=['Low','Medium','High'])
    dm_pvals = {}
    for c1 in ['Low','Medium','High']:
        avg_rps[c1] = round(avgs[c1].mean(),4)
        for c2 in ['Low','Medium','High']:
            if c1 != c2:
                dm_pvals[c1 + ' vs. ' + c2] = Diebold_Mariano_pvalue(avgs[c1], avgs[c2])

    with open(title, 'a') as _file:
        _file.write('Model Complexity\n' + pd.DataFrame(data = np.where((avgs.mean() + t.ppf(1 - (1 - 0.95) / 2, avgs.shape[0] - 1) * avgs.std() / np.sqrt(avgs.shape[0]) < 0.16).values, avg_rps.mean().apply(lambda row: f"{row:.4f}*").values, avg_rps.mean().apply(lambda row: f"{row:.4f}").values), index = ['Low','Medium','High'], columns = ['RPS']).to_html(index=False) + "\nDiebold-Mariano test\n" + pd.DataFrame.from_dict(dm_pvals.items()).rename(columns={0:' ',1:'p-values'}).to_html(index=False))


    #Output Smoothness
    avgs = pd.DataFrame(columns=['Low','Medium','High'])
    avgs['Low'] = rps.iloc[back:,:][cats.loc[cats['Smoothness'] < cats['Smoothness'].describe([0.33, 0.67])['33%'], 'Models'].values].mean(axis=1)
    avgs['Medium'] = rps.iloc[back:,:][cats.loc[(cats['Smoothness'] >= cats['Smoothness'].describe([0.33, 0.67])['33%']) & (cats['Smoothness'] <= cats['Complexity'].describe([0.33, 0.67])['67%']), 'Models'].values].mean(axis=1)
    avgs['High'] = rps.iloc[back:,:][cats.loc[cats['Smoothness'] > cats['Smoothness'].describe([0.33, 0.67])['67%'], 'Models'].values].mean(axis=1)

    avg_rps = pd.DataFrame(index=['Low','Medium','High'])
    dm_pvals = {}
    for c1 in ['Low','Medium','High']:
        avg_rps[c1] = round(avgs[c1].mean(),4)
        for c2 in ['Low','Medium','High']:
            if c1 != c2:
                dm_pvals[c1 + ' vs. ' + c2] = Diebold_Mariano_pvalue(avgs[c1], avgs[c2])

    with open(title, 'a') as _file:
        _file.write('Output Smoothness\n' + pd.DataFrame(data = np.where((avgs.mean() + t.ppf(1 - (1 - 0.95) / 2, avgs.shape[0] - 1) * avgs.std() / np.sqrt(avgs.shape[0]) < 0.16).values, avg_rps.mean().apply(lambda row: f"{row:.4f}*").values, avg_rps.mean().apply(lambda row: f"{row:.4f}").values), index = ['Low','Medium','High'], columns = ['RPS']).to_html(index=False) + "\nDiebold-Mariano test\n" + pd.DataFrame.from_dict(dm_pvals.items()).rename(columns={0:' ',1:'p-values'}).to_html(index=False))


    # Run ensemble techniques
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
        kl_data = kl.iloc[i - back:i,:]

        avgs = rps_data.mean()
        rps_score = (-avgs).rank()
        avgs_kl = kl_data.mean()
        # Caclulate the rank of models whose KL lies in [20%, 80%], by their RPS
        kl_acc_score2 = ((avgs_kl > avgs_kl.describe([0.2,0.8])['20%']) & (avgs_kl < avgs_kl.describe([0.2,0.8])['80%'])) * (-avgs).rank()
        d = rps.iloc[i,:].name


        top_m_names = pd.Index(m_names_ex_ewma)

        truth = pd.DataFrame(columns=["Rank1", "Rank2", "Rank3", "Rank4","Rank5"], data = forecasts.loc[d, ('truth', )].values)

        ############################# ENS_MCS #########################################
        if (i < rps.shape[0] - 1) & ((i % freq == freq - 1) | (i == back)): # For every freq months, recaclulate the top_N models that belong to the MCS
            mcs = MCS(rps_data[top_m_names], size=0.05, seed=RANDOM_SEED)
            mcs.compute()

            #print('Unconditional models')
            unc_models = mcs.pvalues['Pvalue'].nlargest(4).index


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

        ############################### ENS_RPS ##################################
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

        ################################ ENS_SMOOTH_RPS #####################################
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


    #-------------Table 10--------------------
    if universe == 'M6+':
        title = 'results/Table 10 (Panel B).html'
    elif universe == 'M6':
        title = 'results/Table 10 (Panel A).html'
    else:
        title = 'Table 10.html'
    results_top = pd.DataFrame(columns=['ENS_RPS','ENS_MCS','ENS_SMOOTH_RPS'], index=['RPS','W_RPS','pAUC', 'ECE','ACC'])
    results_top['ENS_RPS'] = [np.round(np.mean(score_rps_top), 5), np.round(np.percentile(score_rps_top, 95), 4), np.round(np.mean(pauc_rps_top), 4), np.round(100 * np.mean(ece_rps_top), 2), np.round(np.mean(acc_rps_top), 1)]
    results_top['ENS_MCS'] = [np.round(np.mean(score_unc), 5), np.round(np.percentile(score_unc, 95), 4), np.round(np.mean(pauc_unc), 4), np.round(100 * np.mean(ece_unc), 2), np.round(np.mean(acc_unc), 1)]
    results_top['ENS_SMOOTH_RPS'] = [np.round(np.mean(score_kl_acc2_top), 5), np.round(np.percentile(score_kl_acc2_top, 95), 4), np.round(np.mean(pauc_kl_acc2_top), 4), np.round(100 * np.mean(ece_kl_acc2_top), 2), np.round(np.mean(acc_kl_acc2_top), 1)]

    results_top.T.to_html(title)

#################### FUNCTIONS ####################


if __name__ == '__main__':

    # To run: python Generate_tables_and_figures.py --REPLICATE_PAPER 1
    # OR
    # python Generate_tables_and_figures.py --SAMPLE 'M6' --FILE_NAME 'outputs/Results_M6_with_metrics.xlsx' --TUNING_SAMPLE 12 --FREQ 6

    parser = argparse.ArgumentParser(description='Calculate tables and figures')
    parser.add_argument('--FILE_NAME', nargs='?', type=str, help="The file that contains the forecasts")
    parser.add_argument('--TUNING_SAMPLE', nargs='?', type=int, const=0, default=12, help="The size of the tuning sample. 0 if no tuning sample exists.")
    parser.add_argument('--FREQ', nargs='?', type=int, const=0, default=6)
    parser.add_argument('--REPLICATE_PAPER', nargs='?', type=int, const=1, default=0)
    parser.add_argument('--SAMPLE', nargs='?', type=str, help="M6 for M6 sample, M6+ for M6+ sample and other for other")
    args = parser.parse_args()

    if args.REPLICATE_PAPER:
        calculate_tables('M6', 'outputs/Results_M6_with_metrics.xlsx',  12, 6)
        calculate_tables('M6+', 'outputs/Results_v2_with_metrics.xlsx', 36, 6)
        generate_figure1()
    else:
        calculate_tables(args.SAMPLE, args.FILE_NAME, args.TUNING_SAMPLE, args.FREQ)


    print('\nTask completed...')
    print('\n###################################################################################')
