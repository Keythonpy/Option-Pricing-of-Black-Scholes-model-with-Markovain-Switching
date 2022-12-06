#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Optionpricing import MCexperiment, ECOption, V0, V1, implied_vol
from BSMS import euler_approx_bs, exact_bs
from time import process_time


# In[19]:


if __name__ == '__main__':
    # Define the parameter
    # maturity time
    T = [0.08, 0.25, 0.5, 0.75, 1, 2, 3] #t1
    # starting time
    t0 = 0
    # interest rate
    r = 0.1
    # transition rate lambda
    L01  = 1
    L10 = 1
    
    # Q-matrix from the lambda
    q_mat = np.array([[-L01, L01], [-L10, L10]])
    
    # strike price
    K = 105
    
    # initial stock price
    S0 = 105
    
    # stationary probability pi_{0}
    p0 = 0.5
    
    initial_state = [0, 1]
    
    # drift and volatility
    mu0 = mu1 = r # in risk-neutral measure
    sigma0 = 0.1
    sigma1 = 0.2
    

    para0 = (mu0, sigma0)
    para1 = (mu1, sigma1)

    para = [para0, para1]

    
    # sub-intervals
    stepNos = 100
    
    # time steps
    dt = 1/1000
    
    
    # European call option with classical Black-Scholes formula
    for t1 in T:
        for sigma in [sigma0, sigma1]:                   
            print(f"Classical Black Scoles with sigma = {sigma}, T = {t1}, r = {r}, K = {K}: {ECOption(t1,S0,r,sigma,K)}")
            
    # In risk-neutral measure, close-form solution for B-S with swithching    
    print(f"parameter: (mu0, sigma0) = {para[0]}, (mu1, sigma1) = {para[1]}, r = {r}, K = {K}, S0 = {S0}, pi0 = {p0}, L01 = {L01}, L10 = {L10}")
    for t1 in T:
        
        
        print(f"Close form solution with starting state = 0, T = {t1}: {V0(S0, K, t1, r, sigma0, sigma1, L01, L10, p0)}")
        
        print(f"Close form solution with starting state = 1, T = {t1}: {V1(S0, K, t1, r, sigma0, sigma1, L01, L10, p0)}")
        
    
    # Monte Carlo Simulation with Euler Approximation
    
    M = 300000
    algo = euler_approx_bs
    for t1 in T:
        for i in initial_state:
            print(f"Euler Approximation with starting state {i}, T={t1}: {MCexperiment(M, S0, t0, t1, K, q_mat, i, para, r, algo, stepNos, dt= dt)}")
        
    # Monte Carlo Simulation with Exact method
    
    M = 300000
    algo = exact_bs
    for t1 in T:
        for i in initial_state:
            print(f"Exact method with starting state {i}, T={t1}: {MCexperiment(M, S0, t0, t1, K, q_mat, i, para, r, algo, stepNos, dt= dt)}")
    


# In[ ]:




