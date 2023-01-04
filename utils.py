import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime as dt

def get_peak_frequencies(df:pd.DataFrame, peak_prominence=10**4, show_plot=False):
    # xt = df.index.ravel()
    yt = df.values.ravel()
    fft_output = fft(yt)
    power = np.abs(fft_output)
    freqs = fftfreq(len(fft_output))

    peaks = find_peaks(power[freqs >=0], prominence=peak_prominence)[0] # get most prominent frequencies
    peak_freqs =  freqs[peaks]
    peak_power = power[peaks]
    if show_plot == True:
        plt.stem(peak_freqs, peak_power)
        plt.plot(peak_freqs, peak_power, 'ro')
        plt.title("load profile frequency peaks")
        plt.xlabel("frequency $\left[ \\frac{1}{year} \\right]$")
        plt.ylabel("power")
        plt.show()
    return peak_freqs

def engineer_features(df:pd.DataFrame, peak_freqs:np.array=np.array([])):
    df = df.copy()
    if type(df.index != np.datetime64):
        # print("shits fucked")
        df.index = pd.to_datetime(df.index)
        # print(df.index[0])
    df['day_names'] = df.index.day_name()
    df = df.join(pd.get_dummies(df.index.day_name()).set_index(df.index))

    df['day_of_week'] = df.index.day_of_week
    df['day_of_year'] = df.index.day_of_year
    df['weekend'] = df['day_names'].apply(lambda x: x in ["Saturday", "Sunday"])
    
    cal = calendar()
    sd = df.index.min().date()
    ed = df.index.max().date()
    holidays = cal.holidays(start = sd, end = ed)
    df['holiday'] = pd.to_datetime(df.index.date).isin(holidays)
    for colname in ["weekend", "holiday"]:
        df[colname] = df[colname].astype(int)
    obs_num = df.reset_index().index
    # print(f"peak frequencies:\n{peak_freqs}")
    N = df.shape[0]
    for freq in peak_freqs:
        # df[f'sin_{int(freq * df.shape[0])}_py'] = np.array(np.sin( 1 / df.shape[0] * 2 * np.pi * obs_num* freq))
        df[f'sin_{int(freq * df.shape[0])}_py'] = np.array(np.sin( (2 * np.pi)/N * obs_num / freq))
        # df[f'cos_{int(freq * df.shape[0])}_py'] = np.array(np.cos( 1 / df.shape[0] * 2 * np.pi * obs_num * freq))
        df[f'cos_{int(freq * df.shape[0])}_py'] = np.array(np.cos( (2 * np.pi)/N * obs_num / freq))
        # np.array(np.sin( (2 * np.pi)/N * obs_num/0.05208333 ))



    cols = [colname for colname in df.columns.tolist() if colname != "day_names"] # put load in last column
    return df[cols[1:] + cols[:1]]