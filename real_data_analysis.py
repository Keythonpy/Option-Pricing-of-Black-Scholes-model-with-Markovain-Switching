#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import datetime as dat
import pandas as pd
import pandas_datareader.data as web
# from outliers import smirnov_grubbs as grubbs
from BSMS import euler_approx_bs, exact_bs, sde_ms


# In[3]:


# log return of stock price with day seperated

def log_return(stock_price, days_separated = 1):
    # take logarithm of the prices
    price = np.log(stock_price)
    # to get the log return in specific time interval
    log_returns = []
    for i in range(len(price)):
        if i + days_separated > len(price)-1:
            break
        log_returns.append(price[i + days_separated] - price[i])
    return log_returns


# In[4]:


# drift and volatility of Black-Scholes SDE by maximum likelihood estimation
# Calculate the mean from the log return first
def Mean(x, t=1):
    N = len(x)
    return sum(x) / (N * t)


# calculate the sigma from log return and the mean
def sigma_hat(x, t=1):
    N = len(x)
    M = Mean(x, t)
    Var = np.sqrt(sum((x - M*t)**2) / ((N-1)*t))
    return Var


# calculate the mu
def mu_hat(x, t=1):
    M = Mean(x, t)
    Var  = sigma_hat(x, t)
    mu_hat = M + Var**2 /2
    return mu_hat


# In[5]:


# scrap the S&P500 index value from Yahoo Finance
start = dat.datetime(1970, 1, 1)
end = dat.datetime(2022, 8, 1)


stock_code = '^GSPC'



df = web.DataReader(stock_code, 'yahoo', start, end)


# In[6]:


# define bear and non-bear market
df = df[['Adj Close']].copy()
# Drop percentage
df['percent'] = df['Adj Close']/(df['Adj Close'].cummax()) - 1
# Find out the local maximum value and group it together 
df['localmax'] = ((df['percent'] < 0.) & (df['percent'].shift() == 0.)).cumsum()
# Find out the local minimum value within the local maximum fraction
df['localmin'] = df.groupby('localmax')['percent'].transform('min')
# bear market for drop more than 20% and the smallest in the local maximum fraction
df['bear'] = (df['localmin'] < -0.2) & (df['localmin'] < df.groupby('localmax')['percent'].transform('cummin'))
# count the bear market numbers
df['bearno'] = ((df['bear'] == True) & (df['bear'].shift() == False)).cumsum()

bear_market = df.reset_index().query('bear == True').groupby('bearno')['Date'].agg(['min', 'max'])
print(bear_market)

df['Adj Close'].plot(figsize = (10,7), label='S&P 500 index')
for i, row in bear_market.iterrows():
    if i ==7:
        plt.fill_between(row, df['Adj Close'].max(), alpha=0.3, color='r', label = 'bear market')
    else:
        plt.fill_between(row, df['Adj Close'].max(), alpha=0.3, color='r')
# plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('S&P 500 Index')
plt.title('S&P 500 Index in bear market')
plt.legend()
plt.show()


# In[19]:


#extraction of a proportion of S&P500
start = dat.datetime(2007, 10, 10)
end = dat.datetime(2020, 2, 20)


stock_code = '^GSPC'



df23 = web.DataReader(stock_code, 'yahoo', start, end)


# In[20]:


t0 = 0
t1 = df23.shape[0]
s0 = df23['Adj Close'][0]
dt = 1


# maximum likelihood of all 8 bear market
# L01  = 0.0037
# L10 = 0.00072

# lambda from 2007-10-10 ~ 2020-2-20
L01  = 1/354
L10 = 1/2759

current_state = 0

#drift and volatility from 2007-10-10 ~ 2020-2-20
mu0 = -0.002053
mu1 = 0.0006328
sigma0 = 0.02405
sigma1 = 0.00984


stepNos = 100

para0 = (mu0, sigma0)
para1 = (mu1, sigma1)

para = [para0, para1]

q_mat = np.array([[-L01, L01], [-L10, L10]])



switching = None


# Simulate the Black-Scholes SDE with Markovian switching

for i in range(100):
    
    tmp = np.array([])
    
    #black-scholes model used for markovian switching
    bs = euler_approx_bs(para0)


    bsms = sde_ms(bs, t0, t1, s0, q_mat, current_state, para, stepNos, dt= dt)

    #run gillespie getting the data for Black-Scholes with markovian switching
    
    data = bsms.gillespie()
    
    for d in data:
        tmp = np.concatenate((tmp, d[3]), axis=None)
    
    if i==0:
        switching = tmp[:df23.shape[0]]
    else:
        switching = (switching * i + tmp[:df23.shape[0]]) / (i+1)




# In[21]:


# Simulate the classical Black-Scholes SDE
t0 = 0
t1 = df23.shape[0]
stock = df23['Adj Close'][0]
log_returns = log_return(df23['Adj Close'], 1)
dt = 1
sigma = sigma_hat(log_returns)
mu = mu_hat(log_returns)
para = (mu, sigma)

for i in range(100):
    euler = euler_approx_bs(para)
    k = euler.make_steps(t0, t1, stock, dt)
    s = euler.stock_prices()

    if i==0:
        ex = s[:t1]
    else:
        ex = (ex * i + s[:t1]) / (i+1)
        
        
    euler.change_para(para)


# In[22]:


# plot both simulations with the real S&P index data

fig, ax = plt.subplots(figsize = (15,7))


ax.set_title("Comparison of S&P 500 index and simulations")
ax.plot(df23['Adj Close'], label=r'S&P500')
ax.plot(df23.reset_index()['Date'], switching[:df23.shape[0]], label = 'BS switching')
ax.plot(df23.reset_index()['Date'], ex[:df23.shape[0]], label = 'BS')
ax.legend()
ax.set_ylabel('Index')
ax.set_xlabel('Date')
# plt.savefig("simulation.pdf", format="pdf", bbox_inches="tight")


# In[ ]:




