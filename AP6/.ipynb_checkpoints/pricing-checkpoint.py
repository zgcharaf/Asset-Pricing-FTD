#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[ ]:


#Monte Carlo simulations Rainbow Option Price 75.64


# In[40]:


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# Parameters
num_simulations = 10000  # Number of Monte Carlo simulations
time_steps = 252         # Daily steps over the year
strike_price = 100       # Strike price of the rainbow option
asset_symbols = ['AAPL', 'GOOGL']  # Symbols for the two underlying assets
correlation_matrix = np.array([[1, 0.5], [0.5, 1]])  # Correlation matrix
maturity = 1             # Time to maturity (in years)
risk_free_rate = 0.05    # Risk-free interest rate
dt = maturity / time_steps  # Time step in years

# Download historical price data
data = yf.download(asset_symbols, start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close'].values

# Calculate daily returns as log return from one day to the next
returns = np.log(prices[1:] / prices[:-1])

# Calculate volatilities
volatilities = np.std(returns, axis=0) * np.sqrt(time_steps)

# Prepare the Cholesky decomposition for correlated random variables
cholesky_decomp = cholesky(correlation_matrix)

# Simulate asset prices using Multivariate GBM
np.random.seed(0)
simulated_prices = np.zeros((num_simulations, 2, time_steps + 1))
simulated_prices[:, :, 0] = prices[-1, :]

for i in range(num_simulations):
    for t in range(1, time_steps + 1):
        # Generate correlated random variables
        z = np.dot(cholesky_decomp, np.random.randn(2))
        # Simulate price for each asset
        simulated_prices[i, 0, t] = simulated_prices[i, 0, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[0]**2) * dt +
                                                                          volatilities[0] * np.sqrt(dt) * z[0])
        simulated_prices[i, 1, t] = simulated_prices[i, 1, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[1]**2) * dt +
                                                                          volatilities[1] * np.sqrt(dt) * z[1])

# Calculate the rainbow option payoff for each simulation
max_prices = np.max(simulated_prices, axis=2)
payoffs = np.maximum(np.max(max_prices, axis=1) - strike_price, 0)

# Calculate the option price
option_price = np.exp(-risk_free_rate * maturity) * np.mean(payoffs)

# Calculate the standard error
standard_error = np.std(payoffs) / np.sqrt(num_simulations)

# Calculate a 95% confidence interval
confidence_interval = (option_price - 1.96 * standard_error, option_price + 1.96 * standard_error)

# Plot of simulated asset prices for a subset of simulations
plt.figure(figsize=(10, 6))
for i in range(min(100, num_simulations)):  # Plotting only the first 100 simulations for clarity
    plt.plot(simulated_prices[i, :, :].T, alpha=0.5)
plt.title("Simulated Asset Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()

# Histogram of payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, alpha=0.75, color='blue')
plt.title("Distribution of Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

# Convergence of option price estimate
cumulative_average = np.cumsum(payoffs) / np.arange(1, num_simulations + 1)
plt.figure(figsize=(10, 6))
plt.plot(cumulative_average, color='green')
plt.title("Convergence of Option Price Estimate")
plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Option Price")
plt.axhline(y=option_price, color='red', linestyle='--', label=f'Final Estimate: {option_price:.2f}')
plt.legend()
plt.show()

# Display the results
print(f"Rainbow Option Price: {option_price:.2f}")
print(f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")


# In[ ]:





# In[7]:


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
num_simulations = 10000  # Number of Monte Carlo simulations
time_steps = 252  # Number of time steps per year (daily steps)
strike_price = 100  # Strike price of the rainbow option
asset_symbols = ['AAPL', 'GOOGL']  # Symbols for the two underlying assets
correlation = 0.5  # Correlation between the assets
maturity = 1  # Time to maturity (in years)
risk_free_rate = 0.05  # Risk-free interest rate

# Download historical price data
data = yf.download(asset_symbols, start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close'].values

# Calculate daily returns as log return from one day to the next
returns = np.log(prices[1:] / prices[:-1])

# Calculate volatilities
volatilities = np.std(returns, axis=0) * np.sqrt(time_steps)

# Initialize the array to hold simulated price paths
simulated_prices = np.zeros((num_simulations, time_steps, len(asset_symbols)))
max_payoffs = np.zeros(num_simulations)

# Simulate asset prices
np.random.seed(0)
for i in range(num_simulations):
    asset_prices = np.full((len(asset_symbols),), prices[-1, :])  # Initialize prices to the last known price
    for t in range(time_steps):
        # Generate correlated random numbers
        z1, z2 = np.random.normal(size=2)
        z2 = correlation * z1 + np.sqrt(1 - correlation**2) * z2

        # Simulate price for each asset
        asset_prices[0] *= np.exp((risk_free_rate - 0.5 * volatilities[0]**2) * (maturity / time_steps) +
                                  volatilities[0] * np.sqrt(maturity / time_steps) * z1)
        asset_prices[1] *= np.exp((risk_free_rate - 0.5 * volatilities[1]**2) * (maturity / time_steps) +
                                  volatilities[1] * np.sqrt(maturity / time_steps) * z2)

        # Store the simulated prices
        simulated_prices[i, t, :] = asset_prices

    # Calculate the rainbow option payoff for this simulation
    max_payoffs[i] = np.maximum(np.max(asset_prices) - strike_price, 0)

# Calculate the option price
option_price = np.exp(-risk_free_rate * maturity) * np.mean(max_payoffs)

# Calculate the standard error
standard_error = np.std(max_payoffs) / np.sqrt(num_simulations)

# Calculate a 95% confidence interval
confidence_interval = stats.norm.interval(0.95, loc=option_price, scale=standard_error)

# Plotting the results
# Histogram of max payoffs
plt.figure(figsize=(10, 6))
plt.hist(max_payoffs, bins=50, alpha=0.75, color='blue')
plt.title("Distribution of Max Payoffs")
plt.xlabel("Max Payoff")
plt.ylabel("Frequency")
plt.show()

# Display the results
print(f"Rainbow Option Price: {option_price:.2f}")
print(f"95% Confidence Interval: {confidence_interval}")

# Plot of simulated asset prices for a subset of simulations
plt.figure(figsize=(10, 6))
for i in range(min(100, num_simulations)):  # Plotting only the first 100 simulations for clarity
    plt.plot(range(time_steps), simulated_prices[i, :, 0], alpha=0.5)  # Plotting just one asset for clarity
plt.title("Simulated Asset Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()

# Convergence of option price estimate
payoffs = np.maximum(np.amax(simulated_prices[:, -1, :], axis=1) - strike_price, 0)
cumulative_average = np.cumsum(payoffs) / np.arange(1, num_simulations + 1)
plt.figure(figsize=(10, 6))
plt.plot(cumulative_average, color='green')
plt.title("Convergence of Option Price Estimate")
plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Option Price")
plt.axhline(y=option_price, color='red', linestyle='--', label=f'Final Estimate: {option_price:.2f}')
plt.legend()
plt.show()

# Display the results
print(f"Rainbow Option Price: {option_price:.2f}")
print(f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")


# In[ ]:


#Monte Carlo Simulation and the Binomial Tree Option Pricing Model


# In[10]:


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
num_simulations = 10000  # Number of Monte Carlo simulations
time_steps = 252  # Number of time steps per year (daily steps)
strike_price = 100  # Strike price of the rainbow option
asset_symbols = ['AAPL', 'GOOGL']  # Symbols for the two underlying assets
correlation = 0.5  # Correlation between the assets
maturity = 1  # Time to maturity (in years)
risk_free_rate = 0.05  # Risk-free interest rate

# Download historical price data
data = yf.download(asset_symbols, start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close'].values

# Calculate daily returns as log return from one day to the next
returns = np.log(prices[1:] / prices[:-1])

# Calculate volatilities
volatilities = np.std(returns, axis=0) * np.sqrt(time_steps)


# Calculate daily returns as log return from one day to the next
returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

# Calculate volatilities
volatilities = returns.std() * np.sqrt(time_steps)

# Set up the binomial tree parameters
u1 = np.exp(volatilities[asset_symbols[0]] * np.sqrt(dt))  # Up factor for asset 1
d1 = 1 / u1                                                # Down factor for asset 1
u2 = np.exp(volatilities[asset_symbols[1]] * np.sqrt(dt))  # Up factor for asset 2
d2 = 1 / u2                                                # Down factor for asset 2

# Risk-neutral probabilities
p1 = (np.exp(risk_free_rate * dt) - d1) / (u1 - d1)
p2 = (np.exp(risk_free_rate * dt) - d2) / (u2 - d2)

# Construct the binomial trees for the two assets
price_tree1 = np.zeros((time_steps + 1, time_steps + 1))
price_tree2 = np.zeros((time_steps + 1, time_steps + 1))

# Initial asset prices
price_tree1[0, 0] = data['Adj Close'][asset_symbols[0]][-1]
price_tree2[0, 0] = data['Adj Close'][asset_symbols[1]][-1]

# Populate the binomial trees with simulated end-of-period prices
for i in range(1, time_steps + 1):
    for j in range(i + 1):
        price_tree1[j, i] = price_tree1[0, 0] * (u1 ** (i - j)) * (d1 ** j)
        price_tree2[j, i] = price_tree2[0, 0] * (u2 ** (i - j)) * (d2 ** j)

# Initialize a matrix to store option values at each node
option_values = np.zeros((time_steps + 1, time_steps + 1))

# Calculate option payoffs at maturity
for i in range(time_steps + 1):
    for j in range(time_steps + 1):
        option_values[i, j] = max(0, max(price_tree1[i, time_steps], price_tree2[j, time_steps]) - strike_price)

# Backward induction for option pricing
for step in range(time_steps - 1, -1, -1):
    for i in range(step + 1):
        for j in range(step + 1):
            option_values[i, j] = (p1 * option_values[i + 1, j] + (1 - p1) * option_values[i, j + 1]) * np.exp(-risk_free_rate * dt)

# The value of the option today
option_price = option_values[0, 0]

print(f"The estimated price of the rainbow option is: {option_price:.2f}")


# In[38]:


import numpy as np
import yfinance as yf

# Download historical price data
data = yf.download(['AAPL', 'GOOGL'], start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close']

# Calculate daily returns as log return from one day to the next
returns = np.log(prices / prices.shift(1))

# Calculate volatilities
volatilities = np.std(returns, axis=0) * np.sqrt(252)  # 252 trading days

# Parameters
num_steps = 252  # Number of steps in the binomial tree
maturity = 1  # Time to maturity (in years)
risk_free_rate = 0.05  # Risk-free interest rate
strike_price = 100  # Strike price of the option
dt = maturity / num_steps  # Length of each time step

# Last closing prices of AAPL and GOOGL
current_prices = prices.iloc[-1].values

# Up and down factors for each asset
u_aapl = np.exp(volatilities['AAPL'] * np.sqrt(dt))
d_aapl = 1 / u_aapl
u_googl = np.exp(volatilities['GOOGL'] * np.sqrt(dt))
d_googl = 1 / u_googl

# Risk-neutral probabilities for each asset
p_aapl = (np.exp(risk_free_rate * dt) - d_aapl) / (u_aapl - d_aapl)
p_googl = (np.exp(risk_free_rate * dt) - d_googl) / (u_googl - d_googl)

# Initialize binomial trees for both assets
tree_aapl = np.zeros((num_steps + 1, num_steps + 1))
tree_googl = np.zeros((num_steps + 1, num_steps + 1))

# Populate trees with possible asset prices
for i in range(num_steps + 1):
    for j in range(i + 1):
        tree_aapl[j, i] = current_prices[0] * (u_aapl ** j) * (d_aapl ** (i - j))
        tree_googl[j, i] = current_prices[1] * (u_googl ** j) * (d_googl ** (i - j))

# Function to calculate payoff at each node
def rainbow_option_payoff(price_a, price_b):
    return max(max(price_a, price_b) - strike_price, 0)

# Initialize payoff matrix
option_payoffs = np.zeros((num_steps + 1, num_steps + 1))

# Calculate payoffs at maturity
for i in range(num_steps + 1):
    option_payoffs[i, num_steps] = rainbow_option_payoff(tree_aapl[i, num_steps], tree_googl[i, num_steps])

# Backward induction to value the option
for step in range(num_steps - 1, -1, -1):
    for i in range(step + 1):
        # Average of the two probabilities for simplicity
        p_avg = (p_aapl + p_googl) / 2
        option_payoffs[i, step] = (p_avg * option_payoffs[i, step + 1] + (1 - p_avg) * option_payoffs[i + 1, step + 1]) * np.exp(-risk_free_rate * dt)

# Option price is the value at the root of the tree
option_price = option_payoffs[0, 0]

print(f"Rainbow Option Price (Binomial Model): {option_price:.2f}")


# In[53]:


import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# Parameters for Monte Carlo simulation
num_simulations = 10000  # Number of Monte Carlo simulations
time_steps = 252         # Daily steps over the year
strike_price = 100       # Strike price of the rainbow option
asset_symbols = ['AAPL', 'GOOGL']  # Symbols for the two underlying assets
correlation_matrix = np.array([[1, 0.5], [0.5, 1]])  # Correlation matrix
maturity = 1             # Time to maturity (in years)
risk_free_rate = 0.05    # Risk-free interest rate
dt = maturity / time_steps  # Time step in years

# Download historical price data
data = yf.download(asset_symbols, start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close'].values

# Calculate daily returns and volatilities
returns = np.log(prices[1:] / prices[:-1])
volatilities = np.std(returns, axis=0) * np.sqrt(time_steps)

# Prepare the Cholesky decomposition for correlated random variables
cholesky_decomp = cholesky(correlation_matrix)

# Simulate asset prices using Multivariate GBM
np.random.seed(0)
simulated_prices = np.zeros((num_simulations, 2, time_steps + 1))
simulated_prices[:, :, 0] = prices[-1, :]

for i in range(num_simulations):
    for t in range(1, time_steps + 1):
        z = np.dot(cholesky_decomp, np.random.randn(2))
        simulated_prices[i, 0, t] = simulated_prices[i, 0, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[0]**2) * dt + volatilities[0] * np.sqrt(dt) * z[0])
        simulated_prices[i, 1, t] = simulated_prices[i, 1, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[1]**2) * dt + volatilities[1] * np.sqrt(dt) * z[1])

# Calculate the rainbow option payoff for each simulation
max_prices = np.max(simulated_prices, axis=2)
payoffs = np.maximum(np.max(max_prices, axis=1) - strike_price, 0)

# Calculate the option price
option_price = np.exp(-risk_free_rate * maturity) * np.mean(payoffs)

# Plotting
# Distribution of Maximum Prices at Maturity
max_prices_at_maturity = np.max(simulated_prices[:, :, -1], axis=1)
plt.figure(figsize=(10, 6))
plt.hist(max_prices_at_maturity, bins=50, color='purple', alpha=0.7)
plt.title("Distribution of Maximum Prices at Maturity")
plt.xlabel("Maximum Price")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot of AAPL vs. GOOGL Prices at Maturity
plt.figure(figsize=(10, 6))
plt.scatter(simulated_prices[:, 0, -1], simulated_prices[:, 1, -1], alpha=0.5, color='orange')
plt.title("Scatter Plot of AAPL vs. GOOGL Prices at Maturity")
plt.xlabel("AAPL Price at Maturity")
plt.ylabel("GOOGL Price at Maturity")
plt.grid(True)
plt.show()

# Histogram of Payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, color='green', alpha=0.75)
plt.title("Distribution of Payoffs at Maturity")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

print(f"Rainbow Option Price: {option_price:.2f}")


# In[41]:


pip install scikit-learn numpy


# In[42]:


pip install numpy scikit-learn matplotlib yfinance


# In[44]:


import yfinance as yf
import numpy as np
from scipy.linalg import cholesky

# Parameters
num_simulations = 10000  # Number of Monte Carlo simulations
time_steps = 252         # Daily steps over the year
strike_price = 100       # Strike price of the rainbow option
asset_symbols = ['AAPL', 'GOOGL']  # Symbols for the two underlying assets
correlation_matrix = np.array([[1, 0.5], [0.5, 1]])  # Correlation matrix
maturity = 1             # Time to maturity (in years)
risk_free_rate = 0.05    # Risk-free interest rate
dt = maturity / time_steps  # Time step in years

# Download historical price data
data = yf.download(asset_symbols, start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices
prices = data['Adj Close'].values

# Calculate daily returns as log return from one day to the next
returns = np.log(prices[1:] / prices[:-1])

# Calculate volatilities
volatilities = np.std(returns, axis=0) * np.sqrt(time_steps)

# Prepare the Cholesky decomposition for correlated random variables
cholesky_decomp = cholesky(correlation_matrix)

# Simulate asset prices using Multivariate GBM
np.random.seed(0)
simulated_prices = np.zeros((num_simulations, 2, time_steps + 1))
simulated_prices[:, :, 0] = prices[-1, :]

for i in range(num_simulations):
    for t in range(1, time_steps + 1):
        # Generate correlated random variables
        z = np.dot(cholesky_decomp, np.random.randn(2))
        # Simulate price for each asset
        simulated_prices[i, 0, t] = simulated_prices[i, 0, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[0]**2) * dt +
                                                                          volatilities[0] * np.sqrt(dt) * z[0])
        simulated_prices[i, 1, t] = simulated_prices[i, 1, t-1] * np.exp((risk_free_rate - 0.5 * volatilities[1]**2) * dt +
                                                                          volatilities[1] * np.sqrt(dt) * z[1])

# Calculate the rainbow option payoff for each simulation
max_prices = np.max(simulated_prices, axis=2)
payoffs = np.maximum(np.max(max_prices, axis=1) - strike_price, 0)

# Calculate the option price
option_price = np.exp(-risk_free_rate * maturity) * np.mean(payoffs)

# Display the results
print(f"Rainbow Option Price: {option_price:.2f}")


# In[47]:


import matplotlib.pyplot as plt

# Plot of simulated asset prices for a subset of simulations
plt.figure(figsize=(10, 6))
for i in range(min(100, num_simulations)):  # Plotting only the first 100 simulations for clarity
    plt.plot(simulated_prices[i, 0, :], alpha=0.5, label='AAPL' if i == 0 else "")
    plt.plot(simulated_prices[i, 1, :], alpha=0.5, label='GOOGL' if i == 0 else "")
plt.title("Simulated Asset Price Paths for AAPL and GOOGL")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.show()

# Histogram of payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, alpha=0.75, color='blue')
plt.title("Distribution of Payoffs at Maturity")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()



# In[49]:


# Convergence of option price estimate
cumulative_average = np.cumsum(payoffs) / np.arange(1, num_simulations + 1)
plt.figure(figsize=(10, 6))
plt.plot(cumulative_average, color='green')
plt.title("Convergence of Option Price Estimate")
plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Option Price")
plt.axhline(y=option_price, color='red', linestyle='--', label=f'Final Estimate: {option_price:.2f}')
plt.legend()
plt.show()


# In[50]:


import matplotlib.pyplot as plt

# Path of Maximum Prices
plt.figure(figsize=(10, 6))
max_prices_each_step = np.max(simulated_prices, axis=1)  # Max prices at each time step across all simulations
mean_max_prices = np.mean(max_prices_each_step, axis=0)  # Average of max prices at each time step
plt.plot(mean_max_prices, color='orange')
plt.title("Path of Average Maximum Prices Across All Simulations")
plt.xlabel("Time Steps")
plt.ylabel("Average Maximum Price")
plt.show()

# Final Price Distribution
plt.figure(figsize=(10, 6))
final_prices_aapl = simulated_prices[:, 0, -1]
final_prices_googl = simulated_prices[:, 1, -1]
plt.hist(final_prices_aapl, bins=50, alpha=0.5, label='AAPL', color='blue')
plt.hist(final_prices_googl, bins=50, alpha=0.5, label='GOOGL', color='green')
plt.title("Distribution of Final Prices for AAPL and GOOGL")
plt.xlabel("Final Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Payoff Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(final_prices_aapl, payoffs, alpha=0.5, label='AAPL', color='blue')
plt.scatter(final_prices_googl, payoffs, alpha=0.5, label='GOOGL', color='green')
plt.title("Payoffs vs. Final Prices of AAPL and GOOGL")
plt.xlabel("Final Price")
plt.ylabel("Payoff")
plt.legend()
plt.show()


# In[51]:


import matplotlib.pyplot as plt

# Distribution of Maximum Prices at Maturity
max_prices_at_maturity = np.max(simulated_prices[:, :, -1], axis=1)  # Max price at maturity for each simulation
plt.figure(figsize=(10, 6))
plt.hist(max_prices_at_maturity, bins=50, color='purple', alpha=0.7)
plt.title("Distribution of Maximum Prices at Maturity")
plt.xlabel("Maximum Price")
plt.ylabel("Frequency")
plt.show()




# In[52]:


import matplotlib.pyplot as plt

# Distribution of Maximum Prices at Maturity
max_prices_at_maturity = np.max(simulated_prices[:, :, -1], axis=1)  # Max price at maturity for each simulation
plt.figure(figsize=(10, 6))
plt.hist(max_prices_at_maturity, bins=50, color='purple', alpha=0.7)
plt.title("Distribution of Maximum Prices at Maturity")
plt.xlabel("Maximum Price")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot of AAPL vs. GOOGL Prices at Maturity
plt.figure(figsize=(10, 6))
plt.scatter(simulated_prices[:, 0, -1], simulated_prices[:, 1, -1], alpha=0.5, color='orange')
plt.title("Scatter Plot of AAPL vs. GOOGL Prices at Maturity")
plt.xlabel("AAPL Price at Maturity")
plt.ylabel("GOOGL Price at Maturity")
plt.grid(True)
plt.show()

# Histogram of Payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, color='green', alpha=0.75)
plt.title("Distribution of Payoffs at Maturity")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()


# In[54]:


import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical price data
data = yf.download(['AAPL', 'GOOGL'], start="2022-01-01", end="2023-01-01")

# Extract adjusted closing prices and calculate returns
prices = data['Adj Close']
returns = np.log(prices / prices.shift(1))

# Calculate volatilities
volatilities = np.std(returns.dropna(), axis=0) * np.sqrt(252)  # 252 trading days

# Parameters
num_steps = 252
maturity = 1  # 1 year
risk_free_rate = 0.05
strike_price = 100
dt = maturity / num_steps

# Up and down factors for each asset
u_aapl = np.exp(volatilities['AAPL'] * np.sqrt(dt))
d_aapl = 1 / u_aapl
u_googl = np.exp(volatilities['GOOGL'] * np.sqrt(dt))
d_googl = 1 / u_googl
p = (np.exp(risk_free_rate * dt) - d_aapl) / (u_aapl - d_aapl)  # Assuming same for both

# Initialize binomial trees
tree_aapl = np.zeros((num_steps + 1, num_steps + 1))
tree_googl = np.zeros((num_steps + 1, num_steps + 1))
last_prices = prices.iloc[-1].values

# Populate the trees
for i in range(num_steps + 1):
    for j in range(i + 1):
        tree_aapl[j, i] = last_prices[0] * (u_aapl ** (i - j)) * (d_aapl ** j)
        tree_googl[j, i] = last_prices[1] * (u_googl ** (i - j)) * (d_googl ** j)

# Function for payoff calculation
def rainbow_payoff(price_a, price_b):
    return max(max(price_a, price_b) - strike_price, 0)

# Initialize payoff matrix
payoff_matrix = np.zeros((num_steps + 1, num_steps + 1))

# Calculate payoffs at maturity
for i in range(num_steps + 1):
    for j in range(num_steps + 1):
        payoff_matrix[i, j] = rainbow_payoff(tree_aapl[i, num_steps], tree_googl[j, num_steps])

# Backward induction
for step in range(num_steps - 1, -1, -1):
    for i in range(step + 1):
        for j in range(step + 1):
            payoff_matrix[i, j] = (p * payoff_matrix[i + 1, j] + (1 - p) * payoff_matrix[i, j + 1]) * np.exp(-risk_free_rate * dt)

# Option price is the value at the root of the tree
option_price = payoff_matrix[0, 0]

# Plotting
# Payoff matrix heatmap
plt.figure(figsize=(10, 8))
plt.imshow(payoff_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Payoff Matrix for Rainbow Option")
plt.xlabel("AAPL Step")
plt.ylabel("GOOGL Step")
plt.show()

# Print the option price
print(f"Rainbow Option Price (Binomial Model): {option_price:.2f}")


# In[55]:


# 1. Binomial Tree for AAPL and GOOGL
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Binomial Tree for AAPL")
plt.plot(tree_aapl, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Binomial Tree for GOOGL")
plt.plot(tree_googl, 'o-')
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Distribution of Final Payoffs
final_payoffs = np.maximum(np.maximum(tree_aapl[:, -1], tree_googl[:, -1]) - strike_price, 0)
plt.figure(figsize=(10, 6))
plt.hist(final_payoffs, bins=50, color='blue', alpha=0.7)
plt.title("Distribution of Final Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

# 3. Final Prices vs. Payoff
plt.figure(figsize=(10, 6))
plt.scatter(tree_aapl[:, -1], final_payoffs, alpha=0.5, label='AAPL', color='red')
plt.scatter(tree_googl[:, -1], final_payoffs, alpha=0.5, label='GOOGL', color='green')
plt.title("Final Prices vs. Payoff")
plt.xlabel("Final Asset Price")
plt.ylabel("Payoff")
plt.legend()
plt.show()

