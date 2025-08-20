import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Define ETFs and expected returns & covariance matrix (simulated)
etf_names = ['SPY', 'AGG', 'VEA', 'GLD']

# Annualized expected returns (example)
returns = np.array([0.08, 0.03, 0.06, 0.05])

# Simulated covariance matrix (annualized)
cov_matrix = np.array([
    [0.04, 0.002, 0.004, 0.001],
    [0.002, 0.01, 0.001, 0.0005],
    [0.004, 0.001, 0.03, 0.002],
    [0.001, 0.0005, 0.002, 0.02]
])

# Portfolio weights (example allocation)
weights = np.array([0.4, 0.3, 0.2, 0.1])

# Calculate portfolio expected return and volatility
port_return = np.dot(weights, returns)
port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
print(f'Expected annual return: {port_return:.2%}')
print(f'Expected annual volatility: {port_volatility:.2%}')

# Simulate portfolio performance over 10 years with 1000 simulations
np.random.seed(42)
num_years = 10
num_simulations = 1000

simulated_returns = np.zeros((num_simulations, num_years))

for sim in range(num_simulations):
    yearly_returns = []
    for year in range(num_years):
        # Generate random returns for assets based on multivariate normal distribution
        simulated_asset_returns = np.random.multivariate_normal(returns, cov_matrix)
        portfolio_year_return = np.dot(weights, simulated_asset_returns)
        yearly_returns.append(portfolio_year_return)
    simulated_returns[sim] = yearly_returns

# Calculate mean cumulative returns
cumulative_return = (1 + simulated_returns).cumprod(axis=1) - 1
average_growth = cumulative_return.mean(axis=0)

# Plot average growth of $1 investment over 10 years
plt.figure(figsize=(10,6))
plt.plot(range(1, num_years+1), average_growth, marker='o')
plt.title('Simulated Average Growth of $1 Investment in ETF Portfolio Over 10 Years')
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

