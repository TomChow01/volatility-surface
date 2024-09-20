import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import mean_absolute_percentage_error


def CalculateStrikes(price, window=7):
    if price % 50 > 25:
        atm_strike = round(price / 50) * 50 
    else:
        atm_strike = round(price / 50) * 50
    prices_below = [atm_strike - (i * 50) for i in range(1, window+1)] #itm strikes
    prices_above = [atm_strike + (i * 50) for i in range(1, window+1)] #otm strikes
        
    return atm_strike, prices_below, prices_above


def TimeToExpiry(curr_datetime,expiry) :
    curr_datetime_obj = datetime.strptime(curr_datetime, '%Y-%m-%d %H:%M:%S')
    expiry_obj = datetime.strptime(expiry, '%Y-%m-%d %H:%M:%S')
    difference = expiry_obj - curr_datetime_obj
    days_in_year = 365
    years_difference = difference.total_seconds() / (days_in_year * 24 * 60 * 60)
    return years_difference


def get_last_n_trading_days(n, date, trading_days):
    date_format = '%Y-%m-%d'
    start_date = datetime.strptime(date, date_format)
    date_list = trading_days[trading_days.index(date)-n+1 : trading_days.index(date)+1]
    return date_list


def GetPrev5Min(time_str, lower_bound_str='09:15:00'):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    lower_bound_obj = datetime.strptime(lower_bound_str, '%H:%M:%S')
    previous_5_minutes = []
    for i in range(1, 6):
        new_time = time_obj - timedelta(minutes=i)
        if new_time < lower_bound_obj:
            break
        previous_5_minutes.append(new_time.strftime('%H:%M:%S'))
    previous_5_minutes.reverse()
    
    return previous_5_minutes

def GetPrev30Min(time_str, lower_bound_str='09:15:00'):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    lower_bound_obj = datetime.strptime(lower_bound_str, '%H:%M:%S')
    previous_30_minutes = []
    for i in range(1, 31):
        new_time = time_obj - timedelta(minutes=i)
        if new_time < lower_bound_obj:
            break
        previous_30_minutes.append(new_time.strftime('%H:%M:%S'))
    previous_30_minutes.reverse()
    
    return previous_30_minutes


def convert_to_daily_spot(df, timeframe='24h'):
    # Resample the data into 15-minute intervals and calculate OHLCV
    df_ = df.resample(timeframe, origin='start').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Date': 'first'
    })

    df_.dropna(inplace=True)  # Drop any NaN values that might have been introduced
    return df_

def convert_to_hourly_spot(df, timeframe='1h'):
    # Resample the data into 15-minute intervals and calculate OHLCV
    df_ = df.resample(timeframe, origin='start').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Date': 'first'
    })

    df_.dropna(inplace=True)  # Drop any NaN values that might have been introduced
    return df_

def convert_to_daily_option(df):
    
    # Group by relevant columns except for time
    grouped = df.groupby(['Instrument', 'Option Type', 'Date', 'Expiry', 'Strike price'])

    # Define a function to aggregate each group
    def aggregate_daily(group):
        return pd.Series({
            'Open': group.iloc[0]['Open'],
            'High': group['High'].max(),
            'Low': group['Low'].min(),
            'Close': group.iloc[-1]['Close'],
            'Volume': group['Volume'].sum(),
            'Open Interest': group.iloc[-1]['Open Interest'],
            'index': group['index'].iloc[-1]
        })

    # Apply the aggregation function to each group
    daily_df = grouped.apply(aggregate_daily).reset_index()

    # If needed, reset the index to get back a DataFrame
    # daily_df.head(20)
    return daily_df


def convert_to_hourly_option(df):
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Define the function to align times to start from 09:15:00
    def align_to_hour(dt):
        if dt.minute >= 15:
            return dt.replace(minute=15, second=0, microsecond=0)
        else:
            return (dt - pd.Timedelta(minutes=60)).replace(minute=15, second=0, microsecond=0)

    # Create a new column for the aligned hour
    df['Hour'] = df['Date'].apply(align_to_hour)

    # Group by the new 'Hour' column and the relevant columns except for time
    grouped = df.groupby(['Instrument', 'Option Type', 'Hour', 'Expiry', 'Strike price'])

    # Define a function to aggregate each group
    def aggregate_hourly(group):
        return pd.Series({
            'Open': group.iloc[0]['Open'],
            'High': group['High'].max(),
            'Low': group['Low'].min(),
            'Close': group.iloc[-1]['Close'],
            'Volume': group['Volume'].sum(),
            'Open Interest': group.iloc[-1]['Open Interest'],
            'index': group['index'].iloc[-1]
        })

    # Apply the aggregation function to each group
    hourly_df = grouped.apply(aggregate_hourly).reset_index()

    return hourly_df

# def mean_directional_accuracy(actual, predicted):
#     """
#     Calculates the Mean Directional Accuracy (MDA) for two time series.
    
#     Parameters:
#     actual (array-like): The actual values for the time series.
#     predicted (array-like): The predicted values for the time series.
    
#     Returns:
#     float: The MDA value.
#     """
#     actual = np.array(actual)[:, 5]
#     predicted = np.array(predicted)[:, 5]
#     print('new_mda')
    
#     # calculate the signs of the differences between consecutive values
#     actual_diff = np.diff(actual)
#     actual_signs = np.sign(actual_diff)
#     predicted_diff = np.diff(predicted)
#     predicted_signs = np.sign(predicted_diff)
#     print('Actual vs Predicted', actual_signs, predicted_signs)
    
#     # count the number of times the signs are the same
#     num_correct = np.sum(actual_signs == predicted_signs)
    
#     # calculate the MDA value
#     mda = num_correct / (len(actual) - 1)
    
    # return mda


def mean_directional_accuracy(y_true, y_pred):
    """
    Calculates the mean directional accuracy (MDA) for two time series 

    Args:
        y_true (numpy.ndarray): Array of true labels (0 or 1).
        y_pred (numpy.ndarray): Array of predicted labels (0 or 1).

    Returns:
        float: Mean directional accuracy.
    """
    # print(y_true.shape, y_pred.shape)
    avg_mda = 0
    mda_dict = dict()
    for i in range(len(y_true[0])):
        actual = np.array(y_true)[:, i]
        predicted = np.array(y_pred)[:, i]
        # predicted = np.append(0, np.array(y_pred)[:, i][:-1])
        # print(actual.shape, predicted.shape)
        
        # calculate the signs of the differences between consecutive values
        actual_diff = np.diff(actual)
        actual_signs = np.sign(actual_diff)
        
        predicted_diff = predicted - np.roll(actual, 1)
        predicted_diff = predicted_diff[1:]
        # predicted_diff = np.diff(predicted)
        predicted_signs = np.sign(predicted_diff) 
        # print(actual_signs.shape, predicted_signs.shape)
        
        # count the number of times the signs are the same
        num_correct = np.sum(actual_signs == predicted_signs)
        
        # calculate the MDA value
        mda = num_correct / (len(actual) - 1)
        # print(i, mda)
        mda_dict[f'Strike_{i}'] = mda
        avg_mda += mda
    avg_mda /= len(y_true[0])
    mda_dict['avg_mda'] = avg_mda
    return avg_mda, mda_dict


def my_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # avg_mape = 0
    # for i in range(len(actual[0])):
    #     mape = mean_absolute_percentage_error(actual[:, i], predicted[:, i]) #np.mean(np.abs((actual[:, i] - predicted[:, i]) / actual[:, i])) * 100
    #     avg_mape += mape
    
    mape = mean_absolute_percentage_error(actual, predicted)
    return mape
    
    
    # return np.mean(np.abs((actual - predicted) / actual)) * 100
    



def plot_results(test_iv_df, pred_iv_df,  model:str, scaler=None, n_features=11, save_path='outputs/figures/test.png', noise=None):
    if scaler:
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, n_features))
        y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, n_features))
    
    random_idx = 6 #np.random.randint(2, 13)
    
    plt.figure(figsize=(10, 6.5))
    
    if noise:
        plt.plot(pred_iv_df.iloc[:, random_idx][:500])
        plt.plot(test_iv_df.iloc[:, random_idx][:500])
        # plt.plot(pred_iv_df.iloc[:, random_idx][:500]+ np.random.uniform(-1*noise, 1*noise, size=len(pred_iv_df.iloc[:, random_idx][:500]))) #+ random.choice([-2*noise, 2*noise] 
        # plt.plot(pred_iv_df.iloc[:, random_idx][:500]+ np.random.normal(random.choice(list(range(-2,2))), 1*0.6, size=len(pred_iv_df.iloc[:, random_idx][:500])))
        # plt.plot(random_value_shift_time_series(pred_iv_df.iloc[:, random_idx][:500], n_shifts=20, max_shift_value=1.8) + np.random.normal(0, 1*0.1, size=len(pred_iv_df.iloc[:, random_idx][:500])))
        # plt.plot(test_iv_df.iloc[:, random_idx][:500]) 
    else:
        plt.plot(pred_iv_df.iloc[:, random_idx][:500])
        plt.plot(test_iv_df.iloc[:, random_idx][:500])
    
    plt.xticks([])
    plt.xlabel('Time')
    plt.ylabel('IV')

    
    plt.legend(['y_pred', 'y_true'])
    # plt.title(f"IV Time series for strike {random_idx-1} {model}")
    plt.savefig(save_path.split('.')[0] + '_' + 'IV_Time_Series' + '.png')
    wandb.log({f"IV Time series for strike {random_idx-1} {model}": wandb.Image(plt)})
    # plt.show()
    
    random_idx = 1200 #np.random.randint(0, len(pred_iv_df))
    plt.figure(figsize=(10, 5))
    # print('shape', np.array(pred_iv_df.iloc[random_idx, 2:]).shape)
    if noise:
        plt.plot(np.array(test_iv_df.iloc[random_idx, 2:]).reshape(-1))
        plt.plot(np.array(pred_iv_df.iloc[random_idx, 2:]).reshape(-1))
        # pass
        # plt.plot(np.array(test_iv_df.iloc[random_idx, 2:]).reshape(-1))
        # plt.plot(np.array(pred_iv_df.iloc[random_idx, 2:]).reshape(-1)+ np.random.uniform(-1*(noise/4), 1*(noise/4), size=len(np.array(test_iv_df.iloc[random_idx, 2:]).reshape(-1))))
    else:
        plt.plot(np.array(test_iv_df.iloc[random_idx, 2:]).reshape(-1))
        plt.plot(np.array(pred_iv_df.iloc[random_idx, 2:]).reshape(-1))
        
    plt.legend(['y_true', 'y_pred'])
    plt.title(f"Smile at a random  time stamp {random_idx} {model}")
    plt.savefig(save_path.split('.')[0] + '_' + 'Smile' + '.png')
    wandb.log({f"Smile at a random time stamp {random_idx} {model}": wandb.Image(plt)})
    # plt.show()
    

def plot_surface(test_iv_df, pred_iv_df, model:str, n = 3, save_path='outputs/figures/test.png', noise=None):
        all_exp = test_iv_df['Expiry'].unique()
        rnd_idx = random.choices(list(range(len(all_exp))), k=n)
        expiries = [all_exp[12]]+[all_exp[i] for i in rnd_idx] #random.choices(all_exp, k=n)
        print('EXP: ', expiries)
        
        for i, expiry in enumerate(expiries):
            test_iv_df_exp_1 =test_iv_df[test_iv_df['Expiry'] == expiry]
            pred_iv_df_exp_1 =pred_iv_df[pred_iv_df['Expiry'] == expiry]
            
            expiration_dates = np.arange(0, len(test_iv_df_exp_1), 1) #np.array(test_iv_df_exp_1['Datetime'])
            strikes = np.array([i for i in range(11)])
            values_test = test_iv_df_exp_1.iloc[:, 2:].values
            values_pred = pred_iv_df_exp_1.iloc[:, 2:].values
            # print(expiration_dates.shape, strikes.shape, values_test.shape)

            
            # Create a 3D plot
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')

            # Create a meshgrid for the x and y axes
            X, Y = np.meshgrid(expiration_dates, strikes)
            X = X[:, 1:]
            Y = Y[:, 1:]
            # print(X.shape, Y.shape, values_test.T.shape)
            
            values_test = np.array(values_test.T, dtype=float)[:, 1:]
            values_pred = np.array(values_pred.T, dtype=float)[:, 1:]
            print(X.shape, Y.shape, values_test.shape)
            
            X = np.array(X, dtype=float)
            Y = np.array(Y, dtype=float)

            # Plot the surface
            if noise:
                ax1.plot_surface(X, Y, values_test, cmap='spring', alpha=0.8, label='Actual Implied Volatility')
                ax1.plot_surface(X, Y, values_pred, cmap='summer', alpha=0.8, label='Predicted Implied Volatility')
                # pass
                # ax1.plot_surface(X, Y, values_test , cmap='spring', alpha=0.8, label='Actual Implied Volatility')
                # ax1.plot_surface(X, Y, values_pred + np.random.uniform(-1*noise, 1*noise, size=(len(X), 124)), cmap='summer', alpha=0.8, label='Predicted Implied Volatility')
            else:
                ax1.plot_surface(X, Y, values_test, cmap='spring', alpha=0.8, label='Actual Implied Volatility')
                ax1.plot_surface(X, Y, values_pred, cmap='summer', alpha=0.8, label='Predicted Implied Volatility')
                
            ax1.legend(loc='upper left')

            # Set the plot title and ax1is labels
            ax1.set_title('Implied Volatility Surface')
            ax1.set_xlabel('Expiration Date')
            ax1.set_ylabel('Strike Price')
            ax1.set_zlabel('Implied Volatility')
            wandb.log({f"Implied Volatility Surface {expiry}_{model}": wandb.Image(fig)})

            # Show the plot
            plt.savefig(save_path.split('.')[0] + '_surface_' + str(i) + '.png')
            plt.show()




def random_value_shift_time_series(series, n_shifts=5, max_shift_value=2):
    series_copy = series.copy()
    length = len(series)
    
    for _ in range(n_shifts):
        # Randomly choose start and end for a segment
        start = np.random.randint(0, length // 2)
        end = np.random.randint(start + 1, length)
        
        # Randomly choose a shift value (positive or negative)
        shift_value = np.random.uniform(-max_shift_value, max_shift_value)
        
        # Apply the shift to the values in the selected segment
        series_copy.iloc[start:end] += shift_value
    
    return series_copy
