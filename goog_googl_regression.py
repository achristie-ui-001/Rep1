import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

tickers = ['GOOG', 'GOOGL', 'SPY', 'IWM', 'IWD', 'IWF', 'MTUM', 'QUAL', 'USMV', 'XLK']
data = yf.download(tickers, start='2023-05-01', end='2026-05-01', progress=False, auto_adjust=True)['Close']
returns = data.pct_change().dropna()

factors = pd.DataFrame({
    'Market':   returns['SPY'],
    'Size':     returns['IWM'] - returns['SPY'],
    'Value':    returns['IWD'] - returns['IWF'],
    'Momentum': returns['MTUM'] - returns['SPY'],
    'Quality':  returns['QUAL'] - returns['SPY'],
    'LowVol':   returns['USMV'] - returns['SPY'],
    'Tech':     returns['XLK'] - returns['SPY'],
})

results = {}
for ticker in ['GOOG', 'GOOGL']:
    X = sm.add_constant(factors)
    y = returns[ticker]
    model = sm.OLS(y, X).fit()
    results[ticker] = {
        'params': model.params,
        'resid': model.resid,
        'resid_vol_annual': np.sqrt(model.resid.var() * 252),
        'r_squared': model.rsquared,
    }

print("="*70)
print("FACTOR LOADINGS: GOOG vs GOOGL")
print("="*70)
print(f"{'Factor':<12} {'GOOG':>12} {'GOOGL':>12} {'Diff':>12}")
print("-"*70)
for factor in ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'LowVol', 'Tech']:
    g = results['GOOG']['params'][factor]
    gl = results['GOOGL']['params'][factor]
    print(f"{factor:<12} {g:>12.4f} {gl:>12.4f} {g-gl:>+12.4f}")

print(f"\nSpecific vol GOOG:  {results['GOOG']['resid_vol_annual']*100:.2f}%")
print(f"Specific vol GOOGL: {results['GOOGL']['resid_vol_annual']*100:.2f}%")
print(f"R² GOOG:  {results['GOOG']['r_squared']:.4f}")
print(f"R² GOOGL: {results['GOOGL']['r_squared']:.4f}")

resid_corr = results['GOOG']['resid'].corr(results['GOOGL']['resid'])
print(f"\nResidual correlation: {resid_corr:.4f}")
print(f"Diagonal-D assumes 0. Actual: {resid_corr:.0%}")

a = 0.035
sig2_g = results['GOOG']['resid_vol_annual']**2
sig2_gl = results['GOOGL']['resid_vol_annual']**2
spec_te_diag = np.sqrt(sig2_g * a**2 + sig2_gl * a**2)
cov_spec = resid_corr * np.sqrt(sig2_g * sig2_gl)
spec_te_true = np.sqrt(max(sig2_g * a**2 + sig2_gl * a**2 - 2 * a * a * cov_spec, 0))
print(f"\nPhantom specific TE (diagonal D): {spec_te_diag*10000:.1f} bps")
print(f"True specific TE: {spec_te_true*10000:.1f} bps")
print(f"PHANTOM RISK: {(spec_te_diag-spec_te_true)*10000:.1f} bps")
