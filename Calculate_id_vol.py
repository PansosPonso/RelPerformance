import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#################### FUNCTIONS ####################
def write_or_append_to_excel(df, filename, sheet_name):
    # Check if the file exists
    if os.path.exists(filename):
        # Append to the existing Excel file
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Create a new Excel file and write the DataFrame
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def calc_id_vol(universe, file_name, data_file):
    forecasts_data = pd.read_excel(file_name, sheet_name='forecasts')
    mappings = pd.read_excel(data_file,sheet_name='Tickers')
    closing_prices_df = pd.read_excel(data_file,sheet_name='Close')
    ff = pd.read_excel('F-F_Research_Data_Factors_daily.xlsx').rename(columns={'Unnamed: 0':'Date'}).set_index('Date')

    m_names = [name for name in list(forecasts_data.columns) if "Unnamed" not in name][1:-1]

    tickers = forecasts_data.iloc[2:102,1].values.tolist()
    forecasts = pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([(d, asset) for d in forecasts_data.iloc[2:,0][~pd.isna(forecasts_data.iloc[2:,0])].values for asset in tickers], names=['Date', 'Asset']),
                    columns=pd.MultiIndex.from_tuples([(m, r) for m in [name for name in forecasts_data.columns if "Unnamed" not in name][1:] for r in ["Rank1", "Rank2","Rank3", "Rank4", "Rank5"]], names=['Models', 'Ranks']),
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

    # Caclulate idiosyncratic volatility per asset
    id_vol = {}
    for i in closing_prices_df.columns:
        nonna_sample = ~rets.loc[:,i].isna()
        y = rets.loc[nonna_sample,i].values.reshape(-1, 1) - rets.loc[nonna_sample,'RF'].values.reshape(-1, 1)
        x = rets.loc[nonna_sample,'Mkt-RF'].values.reshape(-1, 1)
        reg = LinearRegression().fit(y, x)
        id_vol[i] = np.sqrt(250) * np.std(y - reg.predict(x))

    results = pd.DataFrame(dict(sorted(id_vol.items(), key=lambda item: item[1])).items())

    write_or_append_to_excel(results, '/outputs/Idiosyncratic_vol.xlsx', universe)

#################### FUNCTIONS ####################


if __name__ == '__main__':

    # To run: python Calculate_id_vol.py --REPLICATE_PAPER 1
    # OR
    # python Calculate_id_vol.py --FILE_NAME 'Results_m6.xlsx' --DATA_FILE 'Data_M6.xlsx'

    parser = argparse.ArgumentParser(description='Calculate tables and figures')
    parser.add_argument('--FILE_NAME', nargs='?', type=str, help="The file that contains the forecasts")
    parser.add_argument('--DATA_FILE', nargs='?', type=str, help="The file that contains the price data")
    parser.add_argument('--REPLICATE_PAPER', nargs='?', type=int, const=1, default=0)
    args = parser.parse_args()

    if args.REPLICATE_PAPER:
        calc_id_vol('M6','/data/Results_M6.xlsx', 'Data_M6.xlsx')
        calc_id_vol('M6+','/data/Results_v2.xlsx', 'Data_v2.xlsx')
    else:
        calc_id_vol('Other', args.FILE_NAME[0], args.DATA_FILE[0])


    print('\nTask completed...')
    print('\n###################################################################################')
