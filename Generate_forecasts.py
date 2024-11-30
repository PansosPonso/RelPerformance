import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import scipy
import heapq
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from tensorflow.keras.utils import to_categorical
from collections import Counter
import argparse

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pyod.models.ecod import ECOD
import warnings
warnings.filterwarnings("ignore")

import datetime
import random
from tqdm import tqdm
import gc
gc.enable()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
import ast
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin

from all_models import copula_model, kde_model, gaussian_mixture_model, deepar_model
from all_models import patchtst_model, flows_model, vae_model, llama_model
from all_models import mnd_model, garch_mnd_model, gan_model, bvar_model, naive_model, ewma_model
from all_models import rforest_model, logistic_model, svm_model, naive_bayes_model, mlp_model, lgbm_model

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



###########################################################################
######################### Helper Functions: Begin #########################
###########################################################################


# Calculate the quintiles for each day
def scores_to_quintiles(x):
    sc = np.zeros((x.shape[0], x.shape[1]), dtype=int)
    for i in range(x.shape[0]):
        sc[i,:] = pd.qcut(x.iloc[i], 5, duplicates='drop', labels=False)
    return sc

# Function for computing RPS score of the competition (Forecasting part). Developed by the organizers (https://github.com/Mcompetitions/M6-methods)
def RPS_calculation(hist_data, submission, asset_no=100):

    if hist_data.shape[0] <= asset_no:
        return np.nan

    asset_id = pd.unique(hist_data.symbol)

    for i in range(len(pd.unique(hist_data.date))):
        if len(hist_data[hist_data.date == pd.unique(hist_data.date)[i]])<len(asset_id):
            for asset in [x for x in asset_id if x not in hist_data[hist_data.date == pd.unique(hist_data.date)[i]].symbol.values]:
                right_price = hist_data[hist_data.symbol==asset].sort_values(by='date')
                right_price = right_price[right_price.date <= pd.unique(hist_data.date)[i]]
                right_price = right_price.price.iloc[-1]
                hist_data = hist_data.append({'date' : pd.unique(hist_data.date)[i],
                                               'symbol' : asset,
                                               'price' : right_price}, ignore_index=True)

    #Compute percentage returns
    asset_id = sorted(asset_id)

    #Compute percentage returns
    returns = pd.DataFrame(columns = ["ID", "Return"])

    min_date = min(hist_data.date)
    max_date = max(hist_data.date)

    for i in range(0,len(asset_id)):
        temp = hist_data.loc[hist_data.symbol==asset_id[i]]

        open_price = float(temp.loc[temp.date==min_date].price)
        close_price = float(temp.loc[temp.date==max_date].price)

        returns = pd.concat([returns, pd.DataFrame({'ID': [temp.symbol.iloc[0]],
                                            'Return': [(close_price - open_price) / open_price]})], ignore_index=True)

    #Define the relevant position of each asset
    ranking = pd.DataFrame(columns=["ID", "Position", "Return"])
    ranking.ID = list(asset_id)
    ranking.Return = returns.Return
    ranking.Position = ranking.Return.rank(method = 'min')

    #Handle Ties
    Series_per_position = pd.DataFrame(columns=["Position","Series", "Rank", "Rank1", "Rank2","Rank3", "Rank4", "Rank5"])
    Series_per_position.Position = list(pd.unique(ranking.Position.sort_values(ascending=True)))
    temp = ranking.Position.value_counts()
    temp = pd.DataFrame(zip(temp.index, temp), columns = ["Rank", "Occurencies"])
    temp = temp.sort_values(by = ["Rank"],ascending=True)
    Series_per_position.Series = list(temp.Occurencies)

    total_ranks = Series_per_position.Position.values[-1]
    for i in range(0,Series_per_position.shape[0]):

        start_p = Series_per_position.Position[i]
        end_p = Series_per_position.Position[i] + Series_per_position.Series[i]
        temp = pd.DataFrame(columns = ["Position","Rank", "Rank1", "Rank2", "Rank3", "Rank4","Rank5"])
        temp.Position = list(range(int(start_p),int(end_p)))

        if(temp.loc[temp.Position.isin(list(range(1,int(0.2*total_ranks+1))))].empty==False):
            temp.loc[temp.Position.isin(list(range(1,int(0.2*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(1,int(0.2*total_ranks+1))))].assign(Rank=1)
            temp.loc[temp.Position.isin(list(range(1,int(0.2*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(1,int(0.2*total_ranks+1))))].assign(Rank1=1.0)

        elif(temp.loc[temp.Position.isin(list(range(int(0.2*total_ranks+1),int(0.4*total_ranks+1))))].empty==False):
            temp.loc[temp.Position.isin(list(range(int(0.2*total_ranks+1),int(0.4*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.2*total_ranks+1),int(0.4*total_ranks+1))))].assign(Rank=2)
            temp.loc[temp.Position.isin(list(range(int(0.2*total_ranks+1),int(0.4*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.2*total_ranks+1),int(0.4*total_ranks+1))))].assign(Rank2=1.0)

        elif(temp.loc[temp.Position.isin(list(range(int(0.4*total_ranks+1),int(0.6*total_ranks+1))))].empty==False):
            temp.loc[temp.Position.isin(list(range(int(0.4*total_ranks+1),int(0.6*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.4*total_ranks+1),int(0.6*total_ranks+1))))].assign(Rank=3)
            temp.loc[temp.Position.isin(list(range(int(0.4*total_ranks+1),int(0.6*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.4*total_ranks+1),int(0.6*total_ranks+1))))].assign(Rank3=1.0)

        elif(temp.loc[temp.Position.isin(list(range(int(0.6*total_ranks+1),int(0.8*total_ranks+1))))].empty==False):
            temp.loc[temp.Position.isin(list(range(int(0.6*total_ranks+1),int(0.8*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.6*total_ranks+1),int(0.8*total_ranks+1))))].assign(Rank=4)
            temp.loc[temp.Position.isin(list(range(int(0.6*total_ranks+1),int(0.8*total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.6*total_ranks+1),int(0.8*total_ranks+1))))].assign(Rank4=1.0)

        elif(temp.loc[temp.Position.isin(list(range(int(0.8*total_ranks+1),int(total_ranks+1))))].empty==False):
            temp.loc[temp.Position.isin(list(range(int(0.8*total_ranks+1),int(total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.8*total_ranks+1),int(total_ranks+1))))].assign(Rank=5)
            temp.loc[temp.Position.isin(list(range(int(0.8*total_ranks+1),int(total_ranks+1))))] = temp.loc[temp.Position.isin(list(range(int(0.8*total_ranks+1),int(total_ranks+1))))].assign(Rank5=1.0)
        temp = temp.fillna(0)
        Series_per_position.iloc[i,2:Series_per_position.shape[1]] = temp.mean(axis = 0).iloc[1:temp.shape[1]]

    Series_per_position = Series_per_position.drop('Series', axis = 1)
    ranking = pd.merge(ranking,Series_per_position, on = "Position")
    ranking = ranking[["ID", "Return", "Position", "Rank", "Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]]
    truth = ranking[["ID", "Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]].set_index('ID')
    truth = truth.reindex(submission['ID']).reset_index(drop=True)
    ranking = ranking.sort_values(["Position"])

    #Evaluate submission
    rps_sub = []
    for aid in asset_id:

        target = np.cumsum(ranking.loc[ranking.ID==aid].iloc[:,4:9].values).tolist()
        frc = np.cumsum(submission.loc[submission.ID==aid].iloc[:,1:6].values).tolist()

        rps_sub.append(np.mean([(a - b)**2 for a, b in zip(target, frc)]))


    submission["RPS"] = rps_sub

    output = {'RPS' : np.round(np.mean(rps_sub), 4),
              'details' : submission,
              'truth' : truth}

    return(output)


def prepare_data(data_universe, data_file):
    # Read asset prices data (as provided by the M6 submission platform)
    asset_data = pd.read_excel(data_universe)

    mappings = pd.read_excel(data_file,sheet_name='Tickers')
    mappings = mappings[['Ticker Factset', 'Ticker']].set_index('Ticker Factset').to_dict()['Ticker']
    mappings['Unnamed: 0'] = 'Date'

    closing_prices_df = pd.read_excel(data_file,sheet_name='Close')
    closing_prices_df = closing_prices_df.rename(columns=mappings).set_index('Date')
    closing_prices_df.index = pd.to_datetime(closing_prices_df.index, format='%d/%m/%Y')
    closing_prices_df = closing_prices_df[closing_prices_df.index <= end_oos_date]
    volume_df = pd.read_excel(data_file,sheet_name='Volume')
    volume_df = volume_df.rename(columns=mappings).set_index('Date')
    volume_df.index = pd.to_datetime(volume_df.index, format='%d/%m/%Y')
    volume_df = volume_df[volume_df.index <= end_oos_date]
    open_p_df = pd.read_excel(data_file,sheet_name='Open')
    open_p_df = open_p_df.rename(columns=mappings).set_index('Date')
    open_p_df.index = pd.to_datetime(open_p_df.index, format='%d/%m/%Y')
    open_p_df = open_p_df[open_p_df.index <= end_oos_date]
    low_df = pd.read_excel(data_file,sheet_name='Low')
    low_df = low_df.rename(columns=mappings).set_index('Date')
    low_df.index = pd.to_datetime(low_df.index, format='%d/%m/%Y')
    low_df = low_df[low_df.index <= end_oos_date]
    high_df = pd.read_excel(data_file,sheet_name='High')
    high_df = high_df.rename(columns=mappings).set_index('Date')
    high_df.index = pd.to_datetime(high_df.index, format='%d/%m/%Y')
    high_df = high_df[high_df.index <= end_oos_date]


    if 'BF-B' in asset_data['symbol'].values:
        asset_data.loc[asset_data['symbol']=='BF-B','symbol'] = 'BF.B'

    tickers = list(closing_prices_df.columns)

    # Keep only business days when the US market was open
    high_df = high_df.resample('B').last()
    low_df = low_df.resample('B').last()
    closing_prices_df = closing_prices_df.resample('B').last()
    open_p_df = open_p_df.resample('B').last()
    volume_df = volume_df.resample('B').last()

    # Fill missing values with previous day's value and calculate returns
    closing_prices_df = closing_prices_df.ffill()
    volume_df = volume_df.ffill()
    high_df = high_df.ffill()
    low_df = low_df.ffill()
    open_p_df = open_p_df.ffill()
    returns_df = closing_prices_df.pct_change(1)

    closing_prices_df = closing_prices_df.iloc[1:,:]
    volume_df = volume_df.iloc[1:,:]
    high_df = high_df.iloc[1:,:]
    low_df = low_df.iloc[1:,:]
    open_p_df = open_p_df.iloc[1:,:]
    returns_df = returns_df.iloc[1:,:]

    # Impute missing values
    imputer = KNNImputer(n_neighbors=10, weights="distance")
    returns_df[tickers] = imputer.fit_transform(returns_df[tickers])

    # Classification for the asset class
    asset_data['AssetClass'] = asset_data['GICS_sector/ETF_type']
    asset_data.loc[asset_data['class'] == 'Stock','AssetClass'] = 'Stocks'

    cleaned_closing_prices = closing_prices_df.copy()

    # Out liars
    # Identify days with strange behavior. Some of them could be days with actual large gains or losses
    strange_days = []
    for f in returns_df.columns:
        clf = ECOD(contamination=0.01)
        clf.fit(returns_df[[f]].fillna(0))
        outlier_ids = np.where(clf.predict(returns_df[[f]]))[0]
        strange_days.extend(returns_df[f].iloc[outlier_ids].index.tolist())

    # Keep only days for which only one stock had a suspiciously large absolute return
    idiosyncratic_days = []
    for i,j in Counter(strange_days).items():
        if j == 1:
            idiosyncratic_days.append(i)

    counter = 0
    # Find and replace outliers in returns
    for f in returns_df.columns:
        clf = ECOD(contamination=0.01)
        clf.fit(returns_df[[f]].fillna(0))
        outlier_ids = np.where(clf.predict(returns_df[[f]]))[0]
        for out in outlier_ids:
            if returns_df[[f]].iloc[out].name in idiosyncratic_days or np.abs(returns_df[[f]].iloc[out].abs().values[0]) > 0.5:
                counter += 1
                if returns_df[f].iloc[out] < 0:
                    returns_df[f].iloc[out] = np.quantile(returns_df[f], 0.01)
                else:
                    returns_df[f].iloc[out] = np.quantile(returns_df[f], 0.99)


    print('\nFixed {} asset-days with outliers...'.format(counter))
    print('\nLast day returns for a sanity check')

    temp_returns = returns_df.copy()
    temp_returns['end'] = temp_returns.index
    temp_returns['start_1mo'] = temp_returns.index - pd.DateOffset(weeks=4)
    temp_returns['start_3mo'] = temp_returns.index - pd.DateOffset(weeks=12)
    temp_returns['days_between_1mo'] = temp_returns.apply(lambda row: returns_df.loc[row['start_1mo']:row['end']].shape[0] - 1, axis=1).astype('int')
    temp_returns['days_between_3mo'] = temp_returns.apply(lambda row: returns_df.loc[row['start_3mo']:row['end']].shape[0] - 1, axis=1).astype('int')
    monthly_returns = pd.DataFrame(columns=tickers, index=temp_returns.index)
    for t in tickers:
        for i,w in zip(range(0, temp_returns.shape[0]), temp_returns['days_between_1mo']):
            monthly_returns[t].iloc[i] = (temp_returns[t].iloc[i-w:i] + 1).prod() - 1

    quarterly_returns = pd.DataFrame(columns=tickers, index=temp_returns.index)
    for t in tickers:
        for i,w in zip(range(0, temp_returns.shape[0]), temp_returns['days_between_3mo']):
            quarterly_returns[t].iloc[i] = (temp_returns[t].iloc[i-w:i] + 1).prod() - 1


    high_low = (high_df - low_df) / cleaned_closing_prices
    high_low = high_low.fillna(0)
    open_close = (cleaned_closing_prices - open_p_df) / open_p_df
    open_close = open_close.fillna(0)
    open_close_gap = (open_p_df - cleaned_closing_prices.shift(1)) / cleaned_closing_prices.shift(1)
    open_close_gap = open_close_gap.fillna(0)
    vol_change = volume_df.pct_change(1)
    vol_change = vol_change.fillna(0)


    print('\n###################################################################################')
    print('\nFeature construction...')
    print('\n###################################################################################')

    # Caclulate various features
    target = monthly_returns.copy()

    feat_0 = monthly_returns.shift(20)
    feat_1 = returns_df.rolling(60).std().shift(20)
    feat_2 = returns_df.rolling(60).max().shift(20)
    feat_3 = np.log(volume_df.ffill().bfill() * cleaned_closing_prices.ffill().bfill() / 1000)
    feat_3 = feat_3.rolling(40).sum().replace([np.inf, -np.inf], np.nan)
    feat_4 = (returns_df.abs() / feat_3).rolling(60).sum().shift(20)
    feat_3 = feat_3.shift(20)
    feat_5 = quarterly_returns.shift(20)
    feat_6 = monthly_returns.shift(20*11)

    y = scores_to_quintiles(returns_df)
    temp = pd.DataFrame(y, columns=tickers, index=returns_df.index)


    # Index with ids for which all rows of the target matrix, contain na
    drop = target.isna().all(axis=1)
    target = target.fillna(0)

    # Caclulate the quintile for each day
    roll_ranks = pd.DataFrame(data=scores_to_quintiles(returns_df.fillna(0)), index=returns_df.index, columns=returns_df.columns)

    # Merge all features together
    train_df = returns_df.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df_temp = target.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['target_return'] = train_df_temp['value']
    train_df_temp = roll_ranks.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['target'] = train_df_temp['value']



    train_df_temp = feat_0.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_0'] = train_df_temp['value']
    train_df_temp = feat_1.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_1'] = train_df_temp['value']
    train_df_temp = feat_2.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_2'] = train_df_temp['value']
    train_df_temp = feat_3.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_3'] = train_df_temp['value']
    train_df_temp = feat_4.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_4'] = train_df_temp['value']
    train_df_temp = feat_5.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_5'] = train_df_temp['value']
    train_df_temp = feat_6.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_6'] = train_df_temp['value']

    train_df = train_df.merge(asset_data, right_on='symbol', left_on='variable')
    train_df.drop(columns=['variable','name'], inplace=True)

    y = scores_to_quintiles(returns_df.shift(20).fillna(0))
    temp = pd.DataFrame(y, columns=tickers, index=returns_df.index)
    q_1 = temp.rolling(20).agg(lambda x: sum(x == 0) / 20).fillna(0.2) # Fill na with 0.2 which is the agnostic mean
    q_2 = temp.rolling(20).agg(lambda x: sum(x == 1) / 20).fillna(0.2)
    q_4 = temp.rolling(20).agg(lambda x: sum(x == 3) / 20).fillna(0.2)
    q_5 = temp.rolling(20).agg(lambda x: sum(x == 4) / 20).fillna(0.2)

    train_df_temp = q_1.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_Rank1'] = train_df_temp['value']
    train_df_temp = q_2.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_Rank2'] = train_df_temp['value']
    train_df_temp = q_4.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_Rank4'] = train_df_temp['value']
    train_df_temp = q_5.reset_index().melt(id_vars=['Date'], value_vars=tickers)
    train_df['feat_Rank5'] = train_df_temp['value']

    # Encode the categorical features
    le_id = LabelEncoder()
    train_df['Stock'] = le_id.fit_transform(train_df['symbol'])

    le = LabelEncoder()
    train_df['AssetClass'] = le.fit_transform(train_df['AssetClass'])

    train_df['target'] = train_df['target'].astype('int32').clip(lower=0)

    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Drop rows with nas and also keep data from 2010 onwards
    train_df = train_df[train_df['Date'].isin(drop[~drop].index)]
    train_df = train_df[train_df['Date'] >= '2010-01-01']

    train = train_df[train_df['Date'] < returns_df.index[-1]]# - pd.offsets.BDay(20)]
    test = train_df.loc[train_df['Date'] == returns_df.index[-1]] # Data for the last day
    # The features we will use
    feats_to_use = [col for col in train if col.startswith('feat_')]

    cleaned_closing_prices = closing_prices_df.astype(np.float32)
    monthly_returns = monthly_returns.astype(np.float32)

    closing_prices_df = closing_prices_df.ffill(axis=0).bfill(axis=0) # replace nan because in backtesting gives errors
    cleaned_closing_prices = cleaned_closing_prices.ffill(axis=0).bfill(axis=0)

    return monthly_returns, returns_df, cleaned_closing_prices, closing_prices_df, train, test, feats_to_use, tickers


#########################################################################
######################### Helper Functions: End #########################
#########################################################################


##############################################################################################
###################################### Evaluate models #######################################
##############################################################################################

class Generate_forecasts:

    def __init__(self, results_file, data_from_previous_run, dates_valid, dates_backtest, monthly_returns, returns_df, cleaned_closing_prices, closing_prices_df, train, test):

        self.configs = [] # Dictionary of configs for each model

        self.results_file = results_file

        self.data_from_previous_run = data_from_previous_run

        self.time = {}
        self.all_forecasts = {}

        self.dates_valid = dates_valid
        self.dates_backtest = dates_backtest
        self.monthly_returns = monthly_returns
        self.returns_df = returns_df
        self.cleaned_closing_prices = cleaned_closing_prices
        self.closing_prices_df = closing_prices_df
        self.train = train
        self.test = test

    def add_model(self, config):
        # if config is not in configs yet add it, or overwrite
        self.configs.append(config)

    def optimize_hyperparameters(self, config):

        def evaluate_opt_results(results, name):
            #Evaluate model on test data using hyperparameters in results

            new_results = results.copy()
            new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

            # Sort with best values on top
            new_results = new_results.sort_values('score', ascending=True).reset_index(drop = True)
            print('The best cross validation score from {} was {:.4f} with parameters {}.'.format(name, new_results.loc[0, 'score'], new_results.loc[0, 'hyperparameters']))

            return new_results.drop(columns='index')

        def score_opt(hyperparameters):

            start = timer()

            config['params'] = hyperparameters
            scores = []
            for d in self.dates_valid:
                eval_train_df = self.train[self.train['Date'].isin(self.returns_df[self.returns_df.index < d - pd.offsets.BusinessDay(n=1)].index)].reset_index(drop=True)
                eval_test_df = self.train[self.train['Date'].isin(self.returns_df[self.returns_df.index == d - pd.offsets.BusinessDay(n=1)].index)].reset_index(drop=True)
                eval_return_df = self.monthly_returns[self.monthly_returns.index < d]
                eval_return_df = eval_return_df.fillna(eval_return_df.rolling(20).mean()).fillna(0)
                eval_return_df = eval_return_df.astype(np.float32)
                eval_monthly_return_df = eval_return_df[eval_return_df.index.isin(pd.date_range(end=eval_return_df.index[-1], freq='4W-FRI', periods=eval_return_df.shape[0] / 20 + 1))]
                eval_closing_df = self.cleaned_closing_prices[self.cleaned_closing_prices.index < d]
                eval_closing_df = eval_closing_df.bfill()

                # These are the data for the following month. We will use them to forecast
                date_range = pd.date_range(start=self.returns_df[self.returns_df.index < d].index[-1].date() + pd.offsets.BusinessDay(n=1), end=self.returns_df[self.returns_df.index < d].index[-1].date() + pd.DateOffset(weeks=4), freq='D')
                d_hist_data = self.closing_prices_df[(self.closing_prices_df.index.isin(date_range))].reset_index().melt(id_vars=['Date'], value_vars=self.closing_prices_df.columns)
                d_hist_data.rename(columns={'variable':'symbol','Date':'date','value':'price'},inplace=True)

                if config['type'] == 'features': # Data for feature based models
                    train_df = eval_train_df
                    test_df = eval_test_df
                elif config['type'] == 'Dreturns': # Return data at the daily frequency
                    train_df = eval_return_df
                    test_df = eval_return_df.iloc[-1:,:]
                elif config['type'] == 'Mreturns': # Return data at the monthly frequency
                    train_df = eval_monthly_return_df
                    test_df = eval_monthly_return_df.iloc[-1:,:]

                # Train model
                config['model'].fit(train_df, hyperparams=config['params'])

                # Forecast next 4-weeks
                forecast = config['model'].forecast(test_df)
                t = RPS_calculation(hist_data = d_hist_data, submission = forecast)
                scores.append(t['RPS'])


            best_score = np.mean(scores)

            run_time = timer() - start

            # Loss must be minimized
            loss = best_score

            # Write to the csv file ('a' means append)
            of_connection = open(OUT_FILE, 'a')
            writer = csv.writer(of_connection)
            writer.writerow([loss, hyperparameters, run_time, best_score])
            of_connection.close()

            # Dictionary with information for evaluation
            return {'loss': loss, 'hyperparameters': hyperparameters,
                    'train_time': run_time, 'status': STATUS_OK}


        if 'hyperparameter_space' in config.keys() and config['optimize_params'] == True:

            print(f'Optimizing parameters for {config["name"]}')

            OUT_FILE = f'{config["name"]}_hyperopt.csv'
            of_connection = open(OUT_FILE, 'w')
            writer = csv.writer(of_connection)

            # Write column names
            headers = ['loss', 'hyperparameters', 'runtime', 'score']
            writer.writerow(headers)
            of_connection.close()

            # Create the algorithm
            tpe_algorithm = tpe.suggest

            # Record results
            trials = Trials()

            MAX_EVALS = 20 * len(config['hyperparameter_space'])

            # Find optimize parameters
            best = fmin(fn=score_opt, space=config['hyperparameter_space'], algo=tpe.suggest, trials=trials, max_evals=MAX_EVALS, early_stop_fn=no_progress_loss(50))

            bayes_results = pd.read_csv(OUT_FILE).sort_values('score', ascending = True).reset_index()
            bayes_params = evaluate_opt_results(bayes_results, name='Hyperopt')

            bayes_best = bayes_params.iloc[bayes_params['score'].idxmin(), :].copy()

            config['optimized_hyperparameters'] = bayes_best['hyperparameters']
            del config['params']

        return config


    def eval_models(self):

        self.all_dates = np.concatenate((self.dates_valid, self.dates_backtest), axis=None)
        
        self.time = pd.DataFrame(
            index=[d for d in self.all_dates],
            columns=[config['name'] for config in self.configs],
            dtype=float
        )

        self.all_forecasts = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(d, asset) for d in self.all_dates for asset in self.monthly_returns.columns], names=['Date', 'Asset']),
            columns=pd.MultiIndex.from_tuples([(m, r) for m in [config['name'] for config in self.configs]+['truth'] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
            dtype=float
        )
        
        calculate_data = 1
        if self.data_from_previous_run and (not pd.read_excel(self.data_from_previous_run, sheet_name='forecasts').empty):
            all_forecasts = pd.read_excel(self.data_from_previous_run, sheet_name='forecasts')
            if not all_forecasts.empty:
                self.all_forecasts.iloc[:,:] = all_forecasts.iloc[2:,2:].values

            if len(self.all_forecasts) == 0:
                self.all_forecasts = pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([(d, asset) for d in self.all_dates for asset in self.monthly_returns.columns], names=['Date', 'Asset']),
                    columns=pd.MultiIndex.from_tuples([(m, r) for m in [config['name'] for config in self.configs]+['truth'] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
                    dtype=float
                )

            if self.all_forecasts.last_valid_index()[0] >= self.all_dates[-1]:
                calculate_data = 0
            else:
                self.all_dates = self.all_dates[self.all_dates > self.all_forecasts.last_valid_index()[0]]

        if calculate_data == 1:

            for d in tqdm(self.all_dates):
                eval_train_df = self.train[self.train['Date'].isin(self.returns_df[self.returns_df.index < d - pd.offsets.BusinessDay(n=1)].index)].reset_index(drop=True)
                eval_test_df = self.train[self.train['Date'].isin(self.returns_df[self.returns_df.index == d - pd.offsets.BusinessDay(n=1)].index)].reset_index(drop=True)
                eval_return_df = self.monthly_returns[self.monthly_returns.index < d]
                eval_return_df = eval_return_df.fillna(eval_return_df.rolling(20).mean()).fillna(0)
                eval_return_df = eval_return_df.astype(np.float32)
                eval_monthly_return_df = eval_return_df[eval_return_df.index.isin(pd.date_range(end=eval_return_df.index[-1], freq='4W-FRI', periods=eval_return_df.shape[0] / 20 + 1))]
                eval_closing_df = self.cleaned_closing_prices[self.cleaned_closing_prices.index < d]
                eval_closing_df = eval_closing_df.bfill()


                # These are the data for the following month. We will use them to forecast
                date_range = pd.date_range(start=self.returns_df[self.returns_df.index < d].index[-1].date() + pd.offsets.BusinessDay(n=1), end=self.returns_df[self.returns_df.index < d].index[-1].date() + pd.DateOffset(weeks=4), freq='D')
                d_hist_data = self.closing_prices_df[(self.closing_prices_df.index.isin(date_range))].reset_index().melt(id_vars=['Date'], value_vars=self.closing_prices_df.columns)
                d_hist_data.rename(columns={'variable':'symbol','Date':'date','value':'price'},inplace=True)

                # Since we have multiple dataframes with training data depending on the model, the training period is caclulated using all dataframes used
                print('\nTraining period: {}-{}'.format(min(eval_closing_df.index[0], eval_train_df['Date'].unique().min(), eval_monthly_return_df.index[0]),
                                                                            max(eval_closing_df.index[-1], eval_train_df['Date'].unique().max(), eval_monthly_return_df.index[-1])))
                print('Evaluation period: {}-{}\n'.format(date_range[0], date_range[-1]))

                for config in self.configs:

                    print(f'Evaluating model {config["name"]}')

                    if config['type'] == 'features':
                        train_df = eval_train_df
                        test_df = eval_test_df
                    elif config['type'] == 'Dreturns':
                        train_df = eval_return_df
                        test_df = eval_return_df.iloc[-1:,:]
                    elif config['type'] == 'Mreturns':
                        train_df = eval_monthly_return_df
                        test_df = eval_monthly_return_df.iloc[-1:,:]

                    start = time.time()

                    # Train model
                    if 'optimized_hyperparameters' in config:
                        config['model'].fit(train_df, hyperparams=config['optimized_hyperparameters'])
                    else:
                        config['model'].fit(train_df)

                    # Forecast next 4-weeks
                    forecast = config['model'].forecast(test_df)
                    t = RPS_calculation(hist_data = d_hist_data, submission = forecast)


                    # Save elapsed time for each model
                    self.time[config['name']].loc[d] = time.time() - start


                    self.all_forecasts.loc[d, (config['name'], )] = forecast[["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]].values

                self.all_forecasts.loc[d, ('truth', )] = t['truth'].values


                with pd.ExcelWriter(self.results_file) as writer:
                    self.all_forecasts.to_excel(writer, sheet_name='forecasts')


    def save_results(self):

        print('\n')
        for config in self.configs:
            print('Average time for model {} is {}\n'.format(config['name'], self.time[config['name']].mean()))

        with pd.ExcelWriter(self.results_file) as writer:
            self.all_forecasts.to_excel(writer, sheet_name='forecasts')


if __name__ == '__main__':

    # To run: python Generate_forecasts.py --MODEL_NAMES LGBM MLP GC MND PatchTST DeepAR GM KDE NB SVM SR RF BVAR VAE NF GAN LagLlama EWMA --SAMPLE 'M6+' --DATA_FROM_PREVIOUS_RUN Results_v2.xlsx

    parser = argparse.ArgumentParser(description='Generate forecasts')
    parser.add_argument('--MODEL_NAMES', nargs='+', type=str, help="Add the models you want to run")
    parser.add_argument('--SAMPLE', nargs='?', type=str, help="M6 for M6 sample, M6+ for M6+ sample and other for other")
    parser.add_argument('--TUNING', nargs='?', type=int, const=0, default=0)
    parser.add_argument('--DATA_FROM_PREVIOUS_RUN', nargs='?', type=str, const=None, default=None)
    args = parser.parse_args()

    print(f'\n\nSelected models: {args.MODEL_NAMES}')

    if args.SAMPLE == 'M6':
        start_oos_date = '2022-03-04'
        end_oos_date = '2023-02-03'
        start_valid_date = '2021-03-05'
        end_valid_date = '2022-02-04'
        data_file = 'data/Data_M6.xlsx'
        data_universe = 'data/Universe_M6.xlsx'
        results_file = 'outputs/Results_M6.xlsx'
    elif args.SAMPLE == 'M6+':
        start_oos_date = '2014-10-17'
        end_oos_date = '2023-12-29'
        start_valid_date = '2011-12-16'
        end_valid_date = '2014-09-19'
        data_file = 'data/Data_v2.xlsx'
        data_universe = 'data/Universe_v2.xlsx'
        results_file = 'outputs/Results_v2.xlsx'
    else:
        start_oos_date = '2014-10-17'
        end_oos_date = '2023-12-29'
        start_valid_date = '2011-12-16'
        end_valid_date = '2014-09-19'
        data_file = 'data/Data_other.xlsx'
        data_universe = 'data/Universe_other.xlsx'
        results_file = 'outputs/Results_other.xlsx'

    start_oos_date = datetime.datetime.strptime(start_oos_date, '%Y-%m-%d')
    end_oos_date = datetime.datetime.strptime(end_oos_date, '%Y-%m-%d')

    monthly_returns, returns_df, cleaned_closing_prices, closing_prices_df, train, test, feats_to_use, tickers = prepare_data(data_universe, data_file)

    models = {
        'RF' : {
            'name': 'RF',
            'model': rforest_model(feats=feats_to_use, tickers=tickers,
                                    params={
                                    'criterion':'entropy',
                                    'max_features':0.5,
                                    'n_estimators':350,
                                    'max_depth':4,
                                    'min_samples_split':6,
                                    'min_samples_leaf':9
                                    }),
            'hyperparameter_space': {
                                    'criterion': hp.choice('criterion',['gini','entropy','log_loss']),
                                    'max_features': hp.choice('max_features',['sqrt','log2',0.5]),
                                    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
                                    'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
                                    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 2)),
                                    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'EWMA' : {
            'name': 'EWMA',
            'model': ewma_model(params={'days':60}),
            'hyperparameter_space': {
                                    'days': scope.int(hp.quniform('days', 20, 120, 10)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'SR' : {
            'name': 'SR',
            'model': logistic_model(feats=feats_to_use, tickers=tickers,
                                    params={
                                    'C':0.6809888523853571
                                    }),
            'hyperparameter_space': {
                                    'C': hp.uniform('C', 0.5, 1.0),
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'SVM' : {
            'name': 'SVM',
            'model': svm_model(feats=feats_to_use, tickers=tickers,
                                params={
                                'C':0.9,
                                'kernel': 'rbf'
                                }),
            'hyperparameter_space': {
                                    'C': hp.uniform('C', 0.5, 1.0),
                                    'kernel': hp.choice('criterion',['linear', 'rbf'])
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'NB' : {
            'name': 'NB',
            'model': naive_bayes_model(feats=feats_to_use, tickers=tickers,
                                        params={
                                                'var_smoothing':1.049774483344398e-05
                                                }
                                    ),
            'hyperparameter_space': {
                                    'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-5), np.log(1e-1)),
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'MLP' : {
            'name': 'MLP',
            'model': mlp_model(feats=feats_to_use, tickers=tickers,
                                params={
                                        'num_hidden_units': 2,
                                        'hidden_unit': 12,
                                        'dropout_rate': 0.5587664122535146,
                                        'label_smoothing': 0.0017927780979064129,
                                        'learning_rate': 0.0005674329417151128,
                                        'epochs':30
                                        }
                             ),
            'hyperparameter_space': {
                                    'num_hidden_units': scope.int(hp.quniform('num_hidden_units', 1, 3, 1)),
                                    'hidden_unit': scope.int(hp.quniform('hidden_unit', 2, 42, 4)),
                                    'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.8),
                                    'label_smoothing': hp.loguniform('label_smoothing', np.log(1e-3), np.log(1e-1)),
                                    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
                                    'epochs': scope.int(hp.quniform('epochs', 10, 50, 10)),
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'LGBM' : {
            'name': 'LGBM',
            'model': lgbm_model(feats=feats_to_use+['AssetClass'], tickers=tickers,
                                params={
                                        'subsample': 0.7,
                                        'subsample_freq': 1,
                                        'learning_rate': 0.02,
                                        'feature_fraction': 0.8,
                                        'linear_tree':True
                                        }
                                ),
            'hyperparameter_space': {
                                    'num_leaves': scope.int(hp.quniform('num_leaves', 16, 512, 16)),
                                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                                    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 55, 5)),
                                    'subsample': hp.uniform('subsample', 0.5, 0.9),
                                    'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.9),
                                    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                                    },
            'type': 'features',
            'optimize_params':args.TUNING
        }
        ,
        'KDE' : {
            'name': 'KDE',
            'model': kde_model(params={'bandwidth':0.10544561027172985}),
            'hyperparameter_space': {
                                    'bandwidth': hp.uniform('bandwidth', 0.1, 1.0),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'GM' : {
            'name': 'GM',
            'model': gaussian_mixture_model(params={'n_components':2}),
            'hyperparameter_space': {
                                    'n_components': scope.int(hp.quniform('n_components', 1, 10, 1)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'NF' : {
            'name': 'NF',
            'model': flows_model(params={
                                        'iter':9000,'hidden_features':2,'num_blocks':3,
                                        'dropout_probability':0.5261707664565186
                                        }
                                ),
            'hyperparameter_space': {
                                    'iter': scope.int(hp.quniform('iter', 5000, 10000, 1000)),
                                    'hidden_features': scope.int(hp.quniform('hidden_features', 2, 6, 2)),
                                    'num_blocks': scope.int(hp.quniform('num_blocks', 1, 3, 1)),
                                    'dropout_probability': hp.uniform('dropout_probability', 0.0, 0.6),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'MND+GARCH' : {
            'name': 'MND+GARCH',
            'model': garch_mnd_model(params={
                                          'lr':1e-2,
                                          'epochs':50
                                          }
                                    ),
            'hyperparameter_space': {
                                    'epochs': scope.int(hp.quniform('epochs', 10, 100, 10)),
                                    'lr': hp.loguniform('lr', np.log(1e-3), np.log(1e-1)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'VAE' : {
            'name': 'VAE',
            'model': vae_model(params={'epochs':800,'batch_size':500,'compress_dims':(128,128),'decompress_dims':(128,128)}),
            'hyperparameter_space': {
                                    'epochs': scope.int(hp.quniform('num_leaves', 200, 1000, 100)),
                                    'compress_dims': hp.choice('compress_dims',[(16,16), (32,32), (64,64), (128,128), (256,256), (512,512)]),
                                    'batch_size': scope.int(hp.quniform('batch_size', 10, 1000, 100)),
                                    },
            'type': 'Dreturns',
            'optimize_params':args.TUNING
        }
        ,
        'LagLlama' : {
            'name': 'Lag-Llama',
            'model': llama_model(params={
                                        'context_length':38,
                                        'lr': 0.03344382235450561,
                                        'n_layer':24,
                                        'batch_size':4,
                                        'max_epochs':30
                                        }
                                ),
            'hyperparameter_space': {
                                    'n_layer': scope.int(hp.quniform('n_layer', 8, 32, 8)),
                                    'context_length': scope.int(hp.quniform('context_length', 2, 512, 2)),
                                    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
                                    'max_epochs': scope.int(hp.quniform('max_epochs', 10, 50, 10)),
                                    'batch_size': scope.int(hp.quniform('batch_size', 1, 12, 1)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'DeepAR' : {
            'name': 'DeepAR',
            'model': deepar_model(params={
                                            'hidden_size':60,
                                            'num_layers':4,
                                            'context_length':2,
                                            'dropout_rate':0.26680398881445383,
                                            'max_epochs':60
                                            }
                                    ),
            'hyperparameter_space': {
                                    'hidden_size': scope.int(hp.quniform('hidden_size', 10, 100, 10)),
                                    'num_layers': scope.int(hp.quniform('num_layers', 1, 5, 1)),
                                    'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.8),
                                    'context_length': scope.int(hp.quniform('context_length', 1, 5, 1)),
                                    'max_epochs': scope.int(hp.quniform('max_epochs', 10, 100, 10)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'PatchTST' : {
            'name': 'PatchTST',
            'model': patchtst_model(params={
                                            'patch_len':8,
                                            'd_model_multiplier':2,
                                            'nhead':8,
                                            'dim_feedforward':240,
                                            'dropout':0.4687597651639158,
                                            'lr':0.001530366537428688,
                                            'max_epochs':90
                                            }
                                    ),
            'hyperparameter_space': {
                                    'patch_len': scope.int(hp.quniform('patch_len', 2, 12, 2)),
                                    'nhead': scope.int(hp.quniform('nhead', 2, 10, 2)),
                                    'd_model_multiplier': scope.int(hp.quniform('d_model_multiplier', 2, 10, 2)),
                                    'dim_feedforward': scope.int(hp.quniform('dim_feedforward', 16, 256, 16)),
                                    'dropout': hp.uniform('dropout', 0.05, 0.8),
                                    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
                                    'max_epochs': scope.int(hp.quniform('max_epochs', 10, 100, 10)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'MND' : {
            'name': 'MND',
            'model': mnd_model(),
            'type': 'Mreturns'
        }
        ,
        'NAIVE' : {
            'name': 'Naive',
            'model': naive_model(tickers=tickers, params={
                                            'out_smooth':8,
                                            }),
            'hyperparameter_space': {
                                    'out_smooth': scope.int(hp.quniform('out_smooth', 2, 8, 1)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'GC' : {
            'name': 'GC',
            'model': copula_model(),
            'type': 'Mreturns'
        }
        ,
        'GAN' : {
            'name': 'GAN',
            'model': gan_model(params={'epochs':200,'batch_size':700,'discriminator_dim':(256,256),'generator_dim':(256,256)}),
            'hyperparameter_space': {
                                    'epochs': scope.int(hp.quniform('epochs', 200, 1000, 100)),
                                    'generator_dim': hp.choice('generator_dim',[(16,16), (32,32), (64,64), (128,128), (256,256), (512,512)]),
                                    'batch_size': scope.int(hp.quniform('batch_size', 10, 1000, 100)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
        ,
        'BVAR' : {
            'name': 'BVAR',
            'model': bvar_model(params={'lags':1}),
            'hyperparameter_space': {
                                    'lags': scope.int(hp.quniform('lags', 1, 5, 1)),
                                    },
            'type': 'Mreturns',
            'optimize_params':args.TUNING
        }
    }

    dates_valid = pd.date_range(start=start_valid_date, end=end_valid_date, freq='4W-MON')
    dates_backtest = pd.date_range(start=start_oos_date, end=end_oos_date, freq='4W-MON')

    eval = Generate_forecasts(results_file, args.DATA_FROM_PREVIOUS_RUN, dates_valid, dates_backtest, monthly_returns, returns_df, cleaned_closing_prices, closing_prices_df, train, test)

    models_configs = []
    for model_name in args.MODEL_NAMES:
        if model_name in models:
            models_configs.append(models[model_name])
        else:
            print(f"Model '{model_name}' not found.")

    for config in models_configs:
        config = eval.optimize_hyperparameters(config)
        eval.add_model(config)


    eval.eval_models()
    eval.save_results()

    gc.collect()

    print('\nTask completed...')
    print('\n###################################################################################')
