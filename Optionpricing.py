#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from scipy.stats import norm
from BSMS import SDE, euler_approx_bs, exact_bs, sde_ms, check_stat_prob
from time import process_time
from math import sqrt


# In[ ]:


# Classical Black-Scholes formula for European call option
def ECOption(T,S0,r,sigma,K):
    # S0: initial stock price at t=0.
    # T: maturity time
    # K: strike price
    # r: interest rate
    # sigma: volatility
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(S0/K)+((r+(1/2)*(sigma**2))*T))
    d2 = d1-sigma*np.sqrt(T) 
    discount = K*np.exp(-r*T)   #discount factor                         
    return S0*norm.cdf(d1) - discount*norm.cdf(d2)


# In[ ]:


# Classical Black-Scholes put option formula
def EPOption(T,S0,r,sigma,K):
    # S0: initial stock price at t=0.
    # T: maturity time
    # K: strike price
    # r: interest rate
    # sigma: volatility
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(S0/K)+((r+(1/2)*(sigma**2))*T)) 
    d2 = d1-sigma*np.sqrt(T) 
    discount = K*np.exp(-r*T)    #discount factor                      
    return discount*norm.cdf(-d2) - S0*norm.cdf(-d1)


# In[ ]:


# Close-form solution
#initial state = 0
def V0(S0, K, T, r, sigma0, sigma1, L0, L1, p0):
    def m(t):
        return np.log(S0) +  (r*T - 0.5*v(t))
    
    def v(t):
        return (sigma0**2 - sigma1**2)*t + sigma1**2*T 
    
    def f(y):
        return (y / (y+K)) * (norm.pdf(np.log(y+K), loc=m(p0*T), scale =np.sqrt(v(p0*T))) * (1-np.exp(-L0*T)) + (
            norm.pdf(np.log(y+K), loc=m(T), scale =np.sqrt(v(T)))) * np.exp(-L0*T))
    
    
    return quad(lambda y: f(y), 0, np.inf)[0] * np.exp(-r*T)

#initial state = 1
def V1(S0, K, T, r, sigma0, sigma1, L0, L1, p0):
    def m(t):
        return np.log(S0) +  (r*T - 0.5*v(t))
    
    def v(t):
        return (sigma0**2 - sigma1**2)*t + sigma1**2*T 
    
    def f(y):
        return (y / (y+K)) * (norm.pdf(np.log(y+K), loc=m(p0*T), scale =np.sqrt(v(p0*T))) * (1-np.exp(-L1*T)) + (
            norm.pdf(np.log(y+K), loc=m(0), scale =np.sqrt(v(0)))) * np.exp(-L1*T))
    return quad(lambda y: f(y), 0, np.inf)[0] * np.exp(-r*T)


# In[ ]:


# Monte Carlo Simulation with B-S switching 
# Payoff function
def Payoff(x,K):
    return max(x-K,0)

# get the asset price data under B-S switching
# calculate the discounted asset price to give one Monte Carlo sample
def MCsample(S0, t0, t1, K, q_mat, current_state, para, r, algo, stepNos, dt):
    # decide which alorithm is used for generate Classical Black-Scholes
    bs = algo(para[current_state])
    # B-S with switching
    bsms = sde_ms(bs, t0, t1, S0, q_mat, current_state, para, stepNos, dt = dt)
    data = bsms.gillespie()
    
    # get the latest price of stock from the simulation
    S = data[-1][3][-1]
    # calculate the discounted payoff
    DiscountPayoff = np.exp(-r*t1)*Payoff(S,K)
    return DiscountPayoff

# Monte Carlo sampling routine for M samples. Returns a tuple of samples.
def MCexperiment(N, S0, t0, t1, K, q_mat, current_state, para, r, algo, stepNos, dt = None):    
    Samples = []
    Exp = [] # store the mean and error
    for _ in range(N):
        Samples.append(MCsample(S0, t0, t1, K, q_mat, current_state, para, r, algo, stepNos, dt))
    
    # Calculate mean and standard error
    a=np.average(Samples)
    b=np.std(Samples,ddof=1)
    Exp.append([a,1.96*b/sqrt(N)])    
    return Exp


# In[ ]:


#implied volatility 
def implied_vol(market, T, S0, r, K):
    iv = 0.01
    price = ECBS(T, S0, r, iv, K)
    while price < market:
        iv += 0.0001
        price = ECBS(T, S0, r, iv, K)
    return float(iv)

