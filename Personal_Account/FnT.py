# Point in time snapshots of PE, SMAs, etc
from matplotlib import ticker
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_data(ticker, start_date):
    return yf.download(ticker, start=start_date)

def moving_average(data, win):
    return data['Close'].rolling(window=win).mean().tail(10)

def plot_moving_average(data, win):
    ma=moving_average(data,win)
    ma.plot(title=f"{ticker} {win}-day Moving Average", figsize=(10,6))
    data['Close'].plot(label='Close Price', alpha=0.5)
    plt.legend()
    plt.show()

def z_score(data, win):
    return (data['Close']-data['Close'].shift(1).rolling(window=win).mean())/data['Close'].shift(1).rolling(window=win).std()

def forwardPE(ticker):
    tick=yf.Ticker(ticker)
