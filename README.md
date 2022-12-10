One of the major deficiencies of Black-Scholes model is the assumption of constant volatility which in fact is varying volatility in real world financial market. This algorithm is to modifiy the classifical Black-Scholes formula with hidden Markov Chain to calculate the fair price of the European options.

BSMS.py: 

1. Classical Black-Scholes formula with Euler Approximation and exact algorithm 
2. Stochastic differential equation with Markovian Switching
3. Verify the model with make_ms_plot() and check_stat_prob()


Optionpricing.py:
1. Option pricing for the Black-scholes in Markov switching with Monte Carlo Simulation
2. Compare with explicit solution V0() and V1()

real_data_analysis.py:
1. scrap the S&P500 index data from yahoo finance
2. identify the hidden Markov chain by classifying the bear market
3. Apply the Black-Scholes SDE with switching 
