import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from py_vollib.black_scholes.implied_volatility import implied_volatility

# import py_vollib.black_scholes_merton.implied_volatility
# import py_vollib_vectorized  

import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from tqdm import tqdm
import yaml

import datetime
# import mibian

from utils import  *


def create_data(data_config_path):
    """
    Creates a dataset of implied volatility values for a given asset.
    Parameters:
        data_config_path (str): The path to the configuration file containing the data settings.
    Returns:
        None
    """
    
    with open(data_config_path) as file:
        data_config = yaml.full_load(file)
        
    start_date =  data_config['start_date'] #'2019-02-11'
    spot_path = data_config['spot_path'] #'Data/NIFTY_SPOT.csv' #'Data/NIFTY_Spot.csv'
    ce_path   = data_config['ce_path'] #'Data/CE.csv'
    pe_path   = data_config['pe_path'] #'Data/PE.csv'

    ## Load Data
    print("Loading Spot Data...")
    df_spot   = pd.read_csv(spot_path)  #pd.read_csv('Data/NIFTY_1.csv', index_col=0, parse_dates=True) #
    df_spot.rename(columns = {'DATE':'Date','TIME':'Time','OPEN':'Open','HIGH':'High','LOW':'Low','CLOSE':'Close','VOLUME':'Volume'},inplace = True)
    df_spot['Datetime'] = df_spot['Date'] + ' ' + df_spot['Time']
    df_spot['Datetime'] = pd.to_datetime(df_spot['Datetime'])
    df_spot.set_index('Datetime',inplace=True)
    df_spot = df_spot[df_spot.index >= pd.to_datetime(start_date + ' ' + '09:15:00')].sort_index()
    
    
    
    df_ce = pd.read_csv(ce_path,index_col=[0])
    df_pe = pd.read_csv(pe_path,index_col=[0])

    # Drop Open, High, Low, Volume, Open Interest
    print("Cleaning Data...")
    cols_to_drop = ['Open', 'High', 'Low', 'Volume', 'Open Interest']
    df_ce.drop(cols_to_drop, axis=1, inplace=True)
    df_pe.drop(cols_to_drop, axis=1, inplace=True)

    
    
    ## Create Datetime Column
    # df_spot['Datetime'] = df_spot['Date'] + ' ' + df_spot['Time']
    df_ce['Datetime'] = df_ce['Date'] + ' ' + df_ce['Time']
    df_pe['Datetime'] = df_pe['Date'] + ' ' + df_pe['Time']

    ## Convert to Datetime
    # df_spot['Datetime'] = pd.to_datetime(df_spot['Datetime'])
    df_ce['Datetime'] = pd.to_datetime(df_ce['Datetime'])
    df_pe['Datetime'] = pd.to_datetime(df_pe['Datetime'])

    ## Set Datetime as index
    # df_spot.set_index('Datetime',inplace=True)
    df_ce.set_index('Datetime',inplace=True)
    df_pe.set_index('Datetime',inplace=True)

    ## Select Date Range (Typically start of option chain datetime)
    # df_spot = df_spot[df_spot.index >= pd.to_datetime(start_date + ' ' + '09:15:00')].sort_index()
    # df_ce = df_ce[df_ce.index >= pd.to_datetime(start_date + ' ' + '09:15:00')].sort_index()
    # df_pe = df_pe[df_pe.index >= pd.to_datetime(start_date + ' ' + '09:15:00')].sort_index()


    df_ce.drop(['index'],axis=1,inplace=True)
    df_pe.drop(['index'],axis=1,inplace=True)
    
    df_ce = df_ce.sort_index()
    df_pe = df_pe.sort_index()
    
    window = int(data_config['smile_window'] // 2)
    sampling_interval = data_config['sampling_interval'] ## in minutes

    columns = ['Datetime', 'Instrument', 'Date', 'Expiry', 'Time', 'Price', 'IV_ITM_5', 'IV_ITM_4', 'IV_ITM_3', 'IV_ITM_2', 'IV_ITM_1', 'IV_ATM', 'IV_OTM_1', 'IV_OTM_2', 'IV_OTM_3', 'IV_OTM_4', 'IV_OTM_5', 'Time to Expiry']

    # Create an empty dataframe with the specified columns
    iv_df = pd.DataFrame(columns=columns)
    
    start_time = datetime.strptime('09:15:00', '%H:%M:%S')
    end_time = datetime.strptime('15:30:00', '%H:%M:%S')
    interval = timedelta(minutes=sampling_interval)

    datetime_list = []
    current_time = start_time
    while current_time <= end_time:
        datetime_list.append(current_time.strftime('%H:%M:%S'))
        current_time += interval
    
    # print(datetime_list)
    # assert 1==2

    for index, row in tqdm(df_spot.iterrows()):
        datetime_index = index
        # print(datetime_index.time().strftime('%H:%M:%S'), datetime_list, type(datetime_list[0]), type(datetime_index.time().strftime('%H:%M:%S')))
        
        # if datetime_index.minute % sampling_interval == 0:
        if datetime_index.time().strftime('%H:%M:%S') in datetime_list:
            # print('datetime: ',datetime_index)
            ivs = []
            # ivs_mibian = []
            # ivs_nr = []
            # ivs_4 = []
            
            date = index.date()
            time = index.time()
            
            price = row['Close']
            
            try:
                ce_data = df_ce.loc[index]
                pe_data = df_pe.loc[index]
            except:
                ce_data = None
                pe_data = None
            
            
            # print('CE Data: ', ce_data, 'PE Data: ', ce_data)
            
            if ce_data is not None and pe_data is not None:
                # ce_data = df_ce.loc[index]
                # option_type = ce_data['Option Type'].values[0]
                assert ce_data.Expiry.values[0] == pe_data.Expiry.values[0]
                expiry = ce_data.Expiry.values[0]
                asset = ce_data.Instrument.values[0]
                atm_strike, prices_below, prices_above = CalculateStrikes(price, window=window)
                # print(atm_strike, prices_below[::-1], prices_above)
                strike_list =  prices_below[::-1] + [atm_strike] + prices_above
                # print('Current Price: ', price)
                # print('Strike List: ', strike_list)
                
                curr_time = datetime_index
                expiry = ce_data.Expiry.values[0]  + ' ' + '15:30:00'
                # print('Current Time %s Expiry %s', str(curr_time), str(expiry))
                
                time_to_exp = TimeToExpiry(str(curr_time),str(expiry))
                

                for strike in strike_list:
                    
                    if ce_data[ce_data['Strike price'] == strike].Close.values.tolist() and pe_data[pe_data['Strike price'] == strike].Close.values.tolist():     
                    
                        try:
                            if strike < price:
                                premium = pe_data[pe_data['Strike price'] == strike].Close.values[0]
                                iv = implied_volatility(premium, price, strike, time_to_exp, r=0.1, flag='p')*100
                            else:
                                premium = ce_data[ce_data['Strike price'] == strike].Close.values[0]
                                iv = implied_volatility(premium, price, strike, time_to_exp, r=0.1, flag='c')*100
                        except:
                            iv = 0
                    
                    else:
                        premium = 0
                        iv = 0
                    
                    # print('Datetime %s Premium %.2f Price %.2f Strike %.2f Time to Expiry %.6f IV %.2f' % (str(index), premium, price, strike, time_to_exp, iv))
                    ivs.append(iv)
                    
                row_values = [index, asset, date, expiry, time, price] + ivs + [time_to_exp] 
                
                # iv_df = iv_df.append(pd.Series(row_values, index=iv_df.columns), ignore_index=True)
                iv_df.loc[len(iv_df)] = row_values

    iv_df.head()
    iv_df.to_csv(data_config['save_path'], index=False)
    print('Data saved to ', data_config['save_path'])
    


if __name__ == '__main__':
    create_data(data_config_path='./data_config.yaml')