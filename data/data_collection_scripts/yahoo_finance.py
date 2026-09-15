import yfinance as yf
import pandas as pd

def get_earnings_info_yahoo(ticker):
    """Get earnings information from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    
    # Get earnings dates
    try:
        earnings_dates = stock.earnings_dates
        print(f"\n{ticker} Earnings Dates:")
        print(earnings_dates.head(10))
    except:
        print(f"No earnings dates for {ticker}")
    
    # Get earnings history
    try:
        earnings_history = stock.earnings_history
        print(f"\n{ticker} Earnings History:")
        print(earnings_history)
    except:
        print(f"No earnings history for {ticker}")
    
    # Get quarterly earnings
    try:
        quarterly = stock.quarterly_earnings
        print(f"\n{ticker} Quarterly Earnings:")
        print(quarterly)
    except:
        print(f"No quarterly earnings for {ticker}")
    
    return stock

# Install: pip install yfinance
ticker = 'AAPL'
stock = get_earnings_info_yahoo(ticker)