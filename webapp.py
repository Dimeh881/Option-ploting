import streamlit as st
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.integrate import quad

def d1f(St,K,t,T,r,sigma):
    '''Black scholes merton d1 function'''
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * (T - t)) / (sigma * math.sqrt(T - t))
    return d1

def dN(x):
    '''Probability density function of standard noormal random variable x'''
    return math.exp(-0.5 * x**2) / math.sqrt(2*math.pi)

def N(d):
    '''Cumulative density function of standard normal random variable x'''
    return quad(lambda x: dN(x), -20, d,limit=50)[0]

def BSM_delta(St,K,t,T,r,sigma,optiont):
        """Black-Scholes-Merton Delta of europiean call option"""
        d1 = d1f(St,K,t,T,r,sigma)
        if optiont == "Call":
            delta = N(d1)
        elif optiont == "Put":
            delta = N(d1) - 1
        return delta
def BSM_gamma(St,K,t,T,r,sigma,optiont):
    '''Black Scholes Merton Gamma of a europian option call'''
    d1 = d1f(St,K,t,T,r,sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T - t))
    return gamma
def BSM_theta(St,K,t,T,r,sigma,optiont):
    '''Black-scholes-merton theta of a europian call option'''
    d1 = d1f(St,K,t,T,r,sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    if optiont == 'Call':
        theta = -(St * dN(d1) * sigma / (2 * math.sqrt(T-t)) + r * K * math.exp(-r * (T-t)) * N(d2))
    elif optiont == 'Put':
        theta = (-St*sigma / 2 * math.sqrt(T-t) * N(d1)) + r*K*math.exp(-r*(T-t)*(1-N(d1)))
    return theta
def BSM_rho(St,K,t,T,r,sigma,optiont):
    '''Black scholes and merton rho of a europian call ption'''
    d1 = d1f(St,K,t,T,r,sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    if optiont == 'Call':
        rho = K * (T - t) * math.exp(-r * (T - t)) * N(d2)
    elif optiont == 'Put':
        rho = (N(d2)-1)*K*(T-t)*math.exp(-r*(T-t))
    return rho 
def BSM_vega(St,K,t,T,r,sigma,optiont):
    '''Black scholes and merton vega of a europian call option'''
    d1 = d1f(St,K,t,T,r,sigma)
    vega = math.sqrt(T-t)*St*dN(d1)
    return vega 
def BSM_option_price(St, K, t, T, r, sigma, option_type='C'):
    '''Calculate the Black-Scholes option price for call or put'''
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    if option_type == 'Call':
        return St * N(d1) - K * math.exp(-r * (T - t)) * N(d2)
    elif option_type == 'Put':
        return K * math.exp(-r * (T - t)) * N(-d2) - St * N(-d1)

def option_payoff(St, K, premium, optiont='Call', position='Long'):
    '''Calculate the payoff of a call or put option'''
    if optiont == 'Call':
        payoff = np.maximum(St - K, 0) - premium
    elif optiont == 'Put':
        payoff = np.maximum(K - St, 0) - premium

    # Adjust for short position
    if position == 'Short':
        payoff = -payoff

    return payoff





st.title('Black-Scholes Merton Model')
if 'position' not in st.session_state:
    st.session_state.position = "Long"
# Input fields
st.sidebar.header('Input Parameters')
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Long Position"):
        st.session_state.position = "Long"

with col2:
    if st.button("Short Position"):
        st.session_state.position = "Short"

# Use the position from session state
position = st.session_state.position

optiont = st.sidebar.radio(
    'Option type',('Call', 'Put'),horizontal=True,index=0)
St = st.sidebar.number_input('Current stock price (St)', value=90.0)
K = st.sidebar.number_input('Strike price (K)', value=120.0)
T = st.sidebar.number_input('Time to maturity (T)', value=1.0)
r = st.sidebar.number_input('Risk-free rate (r)', value=0.03)
sigma = st.sidebar.number_input('Volatility (sigma)', value=0.2)
t = 0.0
premium = BSM_option_price(St, K, t, T, r, sigma, optiont)

strike_min = max(0, K - 60)  # Lower bound: 60 below the selected strike price
strike_max = K + 60          # Upper bound: 60 above the selected strike price
strike_steps = 100   
position = position
# Calculate the option price
option_price = BSM_option_price(St, K, t, T, r, sigma, optiont)

# Separate Payoff chart
st.header("Option Payoff Chart")
stock_prices = np.linspace(0.5 * K, 1.5 * K, 100)

payoffs = option_payoff(stock_prices, K,premium, optiont,position)
fig_payoff = plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoffs, label=f'{optiont} Payoff')
plt.xlabel('Stock Price (St)')
plt.ylabel('Payoff')
plt.title(f'{optiont} Option Payoff')
plt.grid(True)
st.pyplot(fig_payoff)

st.write(f"### The calculated price of the {optiont} option is: **${option_price:.2f}**")


# Select Greek to plot
greek = st.sidebar.selectbox(
    'Select the Greek to plot',
    ('Delta', 'Gamma', 'Theta', 'Vega','Rho',))

if greek == 'Delta':
    greek_calc = BSM_delta
elif greek == 'Gamma':
    greek_calc = BSM_gamma
elif greek == 'Theta':
    greek_calc = BSM_theta
elif greek == 'Rho':
    greek_calc = BSM_rho
elif greek == 'Vega':
    greek_calc = BSM_vega

tlist = np.linspace(0.01,1,25)
klist = np.linspace(strike_min,strike_max,50)
V = np.zeros((len(tlist),len(klist)),dtype=np.float16)
for j in range(len(klist)):
    for i in range(len(tlist)):
        V[i,j] = greek_calc(St,klist[j],t,tlist[i],r,sigma,optiont)

x, y = np.meshgrid(klist, tlist)
st.title('The Options Sensitivity')
# Plotting in Matplotlib
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, V)
ax.set_xlabel('Strike $K$')
ax.set_ylabel('Maturity $T$')
ax.set_zlabel(r'$\Delta(K, T)$')
st.pyplot(fig)

greek_value = greek_calc(St, K, t, T, r, sigma,optiont)
# Display the result
st.write('## Results')
st.write(f'**{greek}**: {round(greek_value,5)}')

if st.button('Recalculate'):
    st.write(f'**Delta**: **{greek_calc}**')

    





