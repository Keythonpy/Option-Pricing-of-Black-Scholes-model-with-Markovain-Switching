#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
import random


# In[2]:


#Create a class for sde simulation
class SDE():
    def __init__(self):
        self.data = [] # for recording stock price and time
        
    def brownian(self, dt):
        return np.random.normal(loc=0, scale=np.sqrt(dt))
    
    def record(self, t, x):
        self.data.append((t,x)) # for recording
        
    def timesteps(self):
        ts, xs = list(zip(*self.data))
        return np.array(ts).T # return all the time
    
    def stock_prices(self):
        ts, xs = list(zip(*self.data))
        return np.array(xs).T # return all the stock price
    
    def make_steps(self):
        raise NotImplementedError 


# In[3]:


# Euler approximation for black-scholes SDE
class euler_approx_bs(SDE):
    def __init__(self, para):
        # mu: drift, which is the interest rate when risk neutral
        # sigma: volatility
        self.mu = para[0] 
        self.sigma = para[1]
        self.data = [] # record the stock price and time
    
    def make_steps(self, t0, t1, x, dt):
        # t0: starting time
        # t1: ending time
        # x: stock price
        # N: number of steps
        t = t0
        while t <= t1:
            self.record(t, x)
            t += dt
            x += self.mu * dt * x + self.sigma * x * self.brownian(dt) 
        return self.data
    
    def change_para(self, para):
        self.data = []
        self.mu = para[0]
        self.sigma = para[1]


# In[4]:


#test the Euler Approximation for Black-Scholes SDE
from time import process_time
t0 = 0
t1 = 1
stock = 100
dt = 1/1000
sigma = 0.05
mu = 0.02
para = (mu, sigma)

fig, ax = plt.subplots(figsize = (20,7))
ax.set_title("Euler Approximation for Black-Scholes SDE")
start = process_time()

euler = euler_approx_bs(para)
e =euler.make_steps(t0, t1, stock, dt)
ax.plot(euler.timesteps(), euler.stock_prices())
euler.change_para(para)
end = process_time()



# In[5]:


# Exact Black-Scholes SDE
class exact_bs(SDE):
    def __init__(self, para):
        # mu: drift, which is the interest rate when risk neutral
        # sigma: volatility
        self.mu = para[0] 
        self.sigma = para[1]
        self.data = [] # record the stock price and time
    
    def make_steps(self, t0, t1, x, dt):
        # t0: starting time
        # t1: ending time
        # x: stock price
        # dt: step interval
        t = t0
        while t <= t1:
            self.record(t, x)
            t += dt
            x = x*np.exp((self.mu - 1/2*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*self.brownian(1))
        return self.data
    
    def change_para(self, para):
        self.data = []
        self.mu = para[0]
        self.sigma = para[1]


# In[6]:


# test the exact Black-Scholes SDE
t0 = 0
t1 = 1
stock = 100
dt = 1/1000
mu = 0.02
sigma = 0.05
para = (mu, sigma)

fig, ax = plt.subplots(figsize = (20,7))
ax.set_title("Exact Black-Scholes SDE")

exact = exact_bs(para)
exact.make_steps(t0, t1, stock, dt)
ax.plot(exact.timesteps(), exact.stock_prices())
exact.change_para(para)


# In[27]:


# Simulate continuous time markov chain with gillespie algorithm
# Varying dt
class sde_ms():

    """
        Stochastic differential equation with Markovian switching 
        "sde" can be plugged in with not only Black-Scholes model but
        also other stochastic differential equation 
    """
    def __init__(self, sde, t0, t1, S0, q_mat, current_state, para, stepNos, dt = None):
        # bs: black-scholes algorithm
        # t0: starting time
        # t1: ending time
        # s0: initial stock price
        # q_mat: Q-matrix with lambdas         
        self.sde = sde
        self.t = t0
        self.t1 = t1
        self.x = S0
        self.current_state = current_state
        self.history = []
        if q_mat.shape[0] != len(para):
            raise Exception("Parameters length not equal to Q-matrix")
        self.q_mat = q_mat 
        self.para = para
        self.dt = dt
        self.stepNos = stepNos
        

    def record(self, holding_t, states, ts, xs):
        self.history.append((holding_t, states, ts, xs))
   
    def next_state(self):
        # pick up the current state value from the Q-matrix
        probs = self.q_mat[self.current_state].copy()
        # set the current state value to be zero in Q-matrix        
        probs[self.current_state] = 0
        # normalised
        probs = probs / sum(probs)
        # return a random sample
        return stats.rv_discrete(values=(range(len(probs)), probs)).rvs()
    
    
    def gillespie(self):
        i = 1
        while self.t < self.t1:            
            # holding time at current state
            L = abs(self.q_mat[self.current_state][self.current_state])        
            holding_t = np.random.exponential(scale=1 / L)
            
            end_t = self.t + holding_t
            
            if end_t > self.t1:
                end_t = self.t1 + 0.001
                holding_t = self.t1 - self.t
            
            delta_t = self.dt if self.dt else holding_t / self.stepNos
                
            
            
            # SDE model at current state with the holding time
            # varying dt
            self.sde.make_steps(self.t, end_t, self.x, delta_t)
            xs = self.sde.stock_prices()
            ts = self.sde.timesteps()
            self.record(holding_t, self.current_state, ts, xs)

            # update current stock price and time
            self.x = xs[-1]
            self.t = ts[-1]
            
            #New parameters after switching
            state = self.next_state()
            self.sde.change_para(self.para[state])           
            self.current_state = state            
        return self.history    


# In[86]:


# test the Euler Approx of Black Scholes SDE with markov switching

#parameters
mu0 = 0.006
sigma0 = 0.05
para0 = (mu0, sigma0)

mu1 = 0.002
sigma1 = 0.1
para1 = (mu1, sigma1)

mu2 = 0.009
sigma2 = 0.2
para2 = (mu2, sigma2)


# para = [para0, para1]

para = [para0, para1, para2]

# time interval
t0 = 0
t1 = 1

# markov state lambda value
lambda01 = 100
lambda02 = 100
lambda10 = 100
lambda12 = 100
lambda20 = 100
lambda21 = 100
# q_mat = np.array([[-lambda01, lambda01], [lambda10, -lambda10]])

lambda00 = lambda01 + lambda02
lambda11 = lambda10 + lambda12
lambda22 = lambda20 + lambda21

q_mat = np.array([[-lambda00, lambda01, lambda02], [lambda10, -lambda11, lambda12], [lambda20, lambda21, -lambda22]])


# initial stock price and state
s0 = 100
current_state = 0

#black-scholes model used for markovian switching
bs = euler_approx_bs(para0)


bsms = sde_ms(bs, t0, t1, s0, q_mat, current_state, para, 1000)

#run gillespie getting the data for Black-Scholes with markovian switching
data = bsms.gillespie()


# In[79]:


def make_ms_plot(data, n, colors = None):
    # n : numbers of states
    
    fig, ax = plt.subplots(1, 1, figsize = (15, 7))
    
    #set up the color if the custom color is missing
    if colors == None:
        colors = plt.cm.jet(np.linspace(0,1,n))
    
    #plot the data with states in different colors
    for i, d in enumerate(data):
        if i == 0:
            ax.plot(d[2], d[3], color = colors[d[1]], label = 'state 0')
        elif i == 1:
            ax.plot(d[2], d[3], color = colors[d[1]], label = 'state 1')
        else:
            ax.plot(d[2], d[3], color = colors[d[1]])
    #plot the data of reduced timesteps with states in different colors    
#     for d in data:
#         reduced_t = [k for i, k in enumerate(d[2]) if i % 2 ==0]
#         reduced_x = [k for i, k in enumerate(d[3]) if i % 2 ==0]
#         ax[1].plot(reduced_t, reduced_x, color = colors[d[1]])
    
    ax.set_title("Black-schole SDE with Markovian switching")
    ax.legend()
    ax.set_ylabel('price')
    ax.set_xlabel('time')
#     plt.savefig("stationary.pdf", format="pdf", bbox_inches="tight")
#     ax[1].set_title("Black-schole SDE for markovian switching with timestep 10*dt")
    
    


# In[6]:


# checking the stationary probability is correct or not
def check_stat_prob(data):
    check = {}
    for d in data:
        time = d[0]
        state = d[1]
        if state not in check:
            check[state] = time
        else:
            check[state] += time
    return check


# In[88]:


check_stat_prob(data)


# In[87]:


nos_state = 3
make_ms_plot(data, nos_state, colors = ['b', 'r', 'k'])


# In[ ]:




