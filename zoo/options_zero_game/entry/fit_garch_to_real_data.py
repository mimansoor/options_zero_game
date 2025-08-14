import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model

def fit_garch_to_ticker(ticker: str, period: str = "60d", interval: str = "5m"):
    """
    Downloads historical intraday and daily price data for a given ticker,
    calculates its returns, fits a GARCH(1,1) model, and analyzes the
    intraday vs. overnight volatility to derive a complete, realistic regime parameter set.
    """
    print(f"--- Fitting GARCH(1,1) model for ticker: {ticker} ---")
    
    # --- Step 1: Get Intraday Data for Volatility Analysis ---
    print(f"Downloading {period} of {interval} intraday data...")
    try:
        stock_data_intraday = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if stock_data_intraday.empty:
            print(f"Error: No intraday data found for ticker {ticker}.")
            return
    except Exception as e:
        print(f"An error occurred while downloading intraday data: {e}")
        return

    # --- Step 2: Prepare Intraday Data & Calculate Volatility Ratio ---
    stock_data_intraday['log_return'] = np.log(stock_data_intraday['Close'] / stock_data_intraday['Close'].shift(1))
    stock_data_intraday['date'] = stock_data_intraday.index.date
    overnight_returns = stock_data_intraday.groupby('date')['log_return'].first().dropna()
    intraday_returns = stock_data_intraday[stock_data_intraday['log_return'].notna() & ~stock_data_intraday.index.isin(overnight_returns.index)]['log_return']

    print(f"Data prepared. Found {len(intraday_returns)} intraday returns and {len(overnight_returns)} overnight gap returns.")

    intraday_vol = intraday_returns.std()
    overnight_vol = overnight_returns.std()
    
    if intraday_vol == 0:
        print("Warning: Intraday volatility is zero. Cannot calculate multiplier.")
        overnight_vol_multiplier = 1.0
    else:
        overnight_vol_multiplier = overnight_vol / intraday_vol

    # --- Step 3: Get Daily Data for GARCH Fitting ---
    print("Downloading 5 years of daily data for GARCH parameter fitting...")
    try:
        daily_data = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if daily_data.empty:
            print(f"Error: No daily data found for ticker {ticker}.")
            return
    except Exception as e:
        print(f"An error occurred while downloading daily data: {e}")
        return

    # <<< FIX: Use the 'Close' column, which is always available.
    daily_log_returns = np.log(daily_data['Close'] / daily_data['Close'].shift(1)).dropna()
    returns_scaled = daily_log_returns * 100
    
    # --- Step 4: Fit the GARCH(1,1) Model ---
    print("Fitting the GARCH(1,1) model...")
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
    print(f"    # Parameters for {ticker} (Daily data: 5y, Intraday data: {period} {interval})")
    print(f"    {{'name': '{ticker}_Real', 'mu': {mu/100:.6f}, 'omega': {omega/10000:.8f}, 'alpha': {alpha:.4f}, 'beta': {beta:.4f}, 'overnight_vol_multiplier': {overnight_vol_multiplier:.2f}}},")
    print("="*60)
    
    if alpha + beta >= 1.0:
        print("\nWARNING: The process is non-stationary (alpha + beta >= 1). The simulation might be explosive.")
    else:
        print(f"\nSUCCESS: The process is stationary (alpha + beta = {alpha+beta:.4f} < 1). These are good parameters.")


if __name__ == "__main__":
    ticker_to_analyze = 'ETH-USD' 
    fit_garch_to_ticker(ticker_to_analyze)

    print("\n" + "="*50 + "\n")

    ticker_to_analyze_2 = 'SOL-USD'
    fit_garch_to_ticker(ticker_to_analyze_2)
