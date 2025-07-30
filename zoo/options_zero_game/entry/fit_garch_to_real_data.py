import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model

def fit_garch_to_ticker(ticker: str, period: str = "60d", interval: str = "5m"):
    """
    Downloads historical intraday price data for a given ticker, calculates its returns,
    fits a GARCH(1,1) model, and analyzes the intraday vs. overnight volatility to
    derive a complete, realistic regime parameter set.

    Args:
        ticker (str): The stock ticker to analyze (e.g., 'TSLA', 'SPY').
        period (str): The period of data to fetch (e.g., "60d" for 60 days).
        interval (str): The data interval (e.g., "5m" for 5 minutes).
    """
    print(f"--- Fitting GARCH(1,1) model for ticker: {ticker} ---")
    
    # --- Step 1: Get Intraday Data ---
    print(f"Downloading {period} of {interval} intraday data...")
    try:
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False)
        if stock_data.empty:
            print(f"Error: No data found for ticker {ticker}. It may not have intraday data available.")
            return
    except Exception as e:
        print(f"An error occurred while downloading data: {e}")
        return

    # --- Step 2: Prepare the Data ---
    stock_data['log_return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    
    # Separate overnight returns from intraday returns
    stock_data['date'] = stock_data.index.date
    overnight_returns = stock_data.groupby('date')['log_return'].first().dropna()
    intraday_returns = stock_data[stock_data['log_return'].notna() & ~stock_data.index.isin(overnight_returns.index)]['log_return']

    print(f"Data prepared. Found {len(intraday_returns)} intraday returns and {len(overnight_returns)} overnight gap returns.")

    # --- Step 3: Analyze Volatility ---
    intraday_vol = intraday_returns.std()
    overnight_vol = overnight_returns.std()
    
    if intraday_vol == 0:
        print("Warning: Intraday volatility is zero. Cannot calculate multiplier.")
        overnight_vol_multiplier = 1.0
    else:
        overnight_vol_multiplier = overnight_vol / intraday_vol

    # --- Step 4: Fit the GARCH(1,1) Model ---
    # We fit the model on the full return series to capture the overall dynamics.
    # We use daily log returns for a more stable GARCH fit.
    daily_data = yf.download(ticker, period="5y", interval="1d", progress=False)
    daily_log_returns = np.log(daily_data['Adj Close'] / daily_data['Adj Close'].shift(1)).dropna()
    returns_scaled = daily_log_returns * 100
    
    print("Fitting the GARCH(1,1) model on 5 years of daily data...")
    model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='Normal')
    results = model.fit(disp='off')

    # --- Step 5: Analyze and Print the Results ---
    print("\n--- GARCH(1,1) Model Fit Results ---")
    print(results.summary())

    params = results.params
    mu = params.get('mu', 0.0)
    omega = params.get('omega', 0.0)
    alpha = params.get('alpha[1]', 0.0)
    beta = params.get('beta[1]', 0.0)

    print("\n" + "="*60)
    print("--- Complete, Copy-Pastable Regime for your Config File ---")
    print("="*60)
    print(f"    # Parameters for {ticker} ({period} {interval} data)")
    print(f"    {{'name': '{ticker}_Real', 'mu': {mu/100:.6f}, 'omega': {omega/10000:.8f}, 'alpha': {alpha:.4f}, 'beta': {beta:.4f}, 'overnight_vol_multiplier': {overnight_vol_multiplier:.2f}}},")
    print("="*60)
    
    if alpha + beta >= 1.0:
        print("\nWARNING: The process is non-stationary (alpha + beta >= 1). The simulation might be explosive.")
    else:
        print(f"\nSUCCESS: The process is stationary (alpha + beta = {alpha+beta:.4f} < 1). These are good parameters.")


if __name__ == "__main__":
    # You can change this ticker to any stock or index you want to analyze!
    ticker_to_analyze = 'TSLA' 
    fit_garch_to_ticker(ticker_to_analyze)

    print("\n" + "="*50 + "\n")

    ticker_to_analyze_2 = 'SPY'
    fit_garch_to_ticker(ticker_to_analyze_2)
