import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.integrate import quad
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from scipy.optimize import least_squares

def d1f(St,K,t,T,r,q,sigma):
    '''Black scholes merton d1 function'''
    d1 = (math.log(St/K) + (r - q + 0.5 * sigma**2) * (T - t)) / (sigma * math.sqrt(T - t))
    return d1

def dN(x):
    '''Probability density function of standard noormal random variable x'''
    return math.exp(-0.5 * x**2) / math.sqrt(2*math.pi)

def N(d):
    '''Cumulative density function of standard normal random variable x'''
    return quad(lambda x: dN(x), -20, d,limit=50)[0]

def BSM_delta(St,K,t,T,r,q,sigma,optiont):
        """Black-Scholes-Merton Delta of europiean call option"""
        d1 = d1f(St,K,t,T,r,q,sigma)
        if optiont == "Call":
            delta = N(d1)
        elif optiont == "Put":
            delta = N(d1) - 1
        return delta
def BSM_gamma(St,K,t,T,r,q,sigma,optiont):
    '''Black Scholes Merton Gamma of a europian option call'''
    d1 = d1f(St,K,t,T,r,q,sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T - t))
    return gamma
def BSM_theta(St, K, t, T, r,q, sigma, optiont):
    '''Black-Scholes-Merton theta of a European option adjusted for dividends'''
    d1 = d1f(St, K, t, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T - t)
    
    if optiont == 'Call':
        theta = -(St * math.exp(-q * (T - t)) * dN(d1) * sigma / (2 * math.sqrt(T - t))) \
                - r * K * math.exp(-r * (T - t)) * N(d2) \
                + q * St * math.exp(-q * (T - t)) * N(d1)
    elif optiont == 'Put':
        theta = -(St * math.exp(-q * (T - t)) * dN(d1) * sigma / (2 * math.sqrt(T - t))) \
                + r * K * math.exp(-r * (T - t)) * N(-d2) \
                - q * St * math.exp(-q * (T - t)) * N(-d1)
    
    return theta

#def BSM_theta(St,K,t,T,r,sigma,optiont):
    '''Black-scholes-merton theta of a europian call option'''
    d1 = d1f(St,K,t,T,r,sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    if optiont == 'Call':
        theta = -(St * dN(d1) * sigma / (2 * math.sqrt(T-t)) + r * K * math.exp(-r * (T-t)) * N(d2))
    elif optiont == 'Put':
        theta = (-St*sigma / 2 * math.sqrt(T-t) * N(d1)) + r*K*math.exp(-r*(T-t)*(1-N(d1)))
    return theta
def BSM_rho(St,K,t,T,r,q,sigma,optiont):
    '''Black scholes and merton rho of a europian call ption'''
    d1 = d1f(St,K,t,T,r,q,sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    if optiont == 'Call':
        rho = K * (T - t) * math.exp(-r * (T - t)) * N(d2)
    elif optiont == 'Put':
        rho = (N(d2)-1)*K*(T-t)*math.exp(-r*(T-t))
    return rho 
def BSM_vega(St,K,t,T,r,q,sigma,optiont):
    '''Black scholes and merton vega of a europian call option'''
    d1 = d1f(St,K,t,T,r,q,sigma)
    vega = math.sqrt(T-t)*St*dN(d1)
    return vega 
def BSM_option_price(St, K, t, T, r, sigma, option_type='Call', q=0.0):
    """Calculate the Black-Scholes option price for a call or put option considering dividends"""
    d1 = (math.log(St/K) + (r - q + 0.5 * sigma**2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    
    if option_type == 'Call':
        price = St * math.exp(-q * (T - t)) * N(d1) - K * math.exp(-r * (T - t)) * N(d2)
    elif option_type == 'Put':
        price = K * math.exp(-r * (T - t)) * N(-d2) - St * math.exp(-q * (T - t)) * N(-d1)
    
    return price

# def BSM_option_price(St, K, t, T, r,q, sigma, option_type='C'):
    '''Calculate the Black-Scholes option price for call or put'''
    d1 = d1f(St, K, t, T, r,q, sigma)
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

def diff(sigma, St, K, t, T, r, market_price, option_type):
    '''Calculate the difference between market price and Black-Scholes price'''
    return np.abs(market_price - BSM_option_price(St, K, t, T, r, sigma, option_type))


st.title('Black-Scholes Merton Model')
st.markdown("""
This application is designed to calculate and visualize various options metrics using the Black-Scholes Merton Model. 
You can explore different strike prices, expiration dates, and option types to analyze the sensitivity of options. 

**Developed by [Benomar Mehdi](https://www.linkedin.com/in/mehdi-benomar-671ba7295/)**

""")



if 'position' not in st.session_state:
    st.session_state.position = "Long"
# Input fields
st.sidebar.header('Input Parameters')
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Long Position"):
        st.session_state.position = "Long"

with col2:
    if st.button("Short Position"):
        st.session_state.position = "Short"
optiont = st.sidebar.radio('Option type', ('Call', 'Put'), horizontal=True, index=0)
stock = yf.Ticker(ticker)
stock_price = stock.history(period='1d')['Close'][0]  # Get the latest closing price
St = float(stock_price)
  # Initial guess for volatility

# Fetch available expiration dates for the selected ticker
expirations = stock.options

# Allow the user to select the expiration date
expiration_date = st.sidebar.selectbox("Select Expiration Date", expirations)

# Fetch the option chain for the selected expiration date
option_chain = stock.option_chain(expiration_date)
options = option_chain.calls if optiont == 'Call' else option_chain.puts
strike_prices = options['strike'].tolist()
K = float(st.sidebar.selectbox('Strike price (K)', strike_prices))
r = st.sidebar.number_input('Risk-free rate (r)', value=4.00, format="%.2f", step=0.10) / 100.0
#r = float(st.sidebar.number_input('Risk-free rate (r)', value=0.040,format="%.3f",step=0.001))
q = st.sidebar.number_input('Dividend Yield (q)', value=2.00, format="%.2f", step=0.10) / 100.0
#q = float(st.sidebar.number_input('Dividend Yield (q)', value=0.0, format="%.4f", step=0.001))
sigma = st.sidebar.number_input('Volatility (sigma)', value=23.73, format="%.2f", step=0.10) / 100.0
#sigma = float(st.sidebar.number_input('Volatility (sigma)', value=0.25,format="%.4f",step=0.001))
t = 0.0
initial_sigma = sigma


# Filter for the specific strike price



matching_options = options[(options['strike'] == K)]

if not matching_options.empty:
    T = (pd.to_datetime(expiration_date) - pd.to_datetime('today')).days / 360.0  # Time to maturity in years
    real_market_price = matching_options.iloc[0]['lastPrice']

    # Calculate the Black-Scholes option price
    option_price = BSM_option_price(St, K, t, T, r, sigma, optiont)

    st.subheader(f"{ticker} Options Tear Sheet")

# Organize the layout into two columns
col3, col5 = st.columns(2)

with col3:
    st.subheader("")
    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Current Stock Price:** ${St:.2f}")
    st.write(f"**Option Type:** {optiont}")
    st.write(f"**Strike Price:** ${K:.2f}")
    st.write(f"**Expiration Date:** {expiration_date}")
    st.write(f"**Time to Maturity:** {T:.2f} years")

with col5:
    
    st.subheader("")
    st.markdown(f"**Real Market Price:** <span style='font-size: 20px;'>${real_market_price:.2f}</span></p>", unsafe_allow_html=True)
    st.markdown(f"**Calculated Black-Scholes Price:** <span style='font-size: 20px;'>${option_price:.2f}</span></p>", unsafe_allow_html=True)
    st.markdown(f"**Difference:** <span style='font-size: 20px;'>${option_price - real_market_price:.2f}</span></p>", unsafe_allow_html=True)


if st.button("**---Adjust Volatility to Match Market Price---**"):
    initial_guess = 1
    result = least_squares(diff, initial_guess, args=(St, K, t, T, r, real_market_price, optiont))
    adjusted_sigma = result.x[0]
    st.sidebar.write(f"Estimated Implied Volatility: {adjusted_sigma * 100:.3f}%")
    sigma = adjusted_sigma  # Update sigma with the optimized value
    option_price = BSM_option_price(St, K, t, T, r, sigma, optiont)  # Recalculate option price with optimized sigma

position = st.session_state.position
premium = BSM_option_price(St, K, t, T, r, sigma, optiont)
strike_min = max(0, K - 60)  # Lower bound: 60 below the selected strike price
strike_max = K + 60          # Upper bound: 60 above the selected strike price
strike_steps = 100   
position = position
# Calculate the option price




# Separate Payoff chart
st.header("Option Payoff Chart")
stock_prices = np.linspace(0.5 * K, 1.5 * K, 100)

payoffs = option_payoff(stock_prices, K,premium, optiont,position)

fig_payoff = plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoffs, label=f'{optiont} Payoff')
plt.xlabel('Stock Price (St)')
plt.ylabel('Payoff')
plt.title(f'{position,optiont} Option Payoff')
plt.grid(True)
st.pyplot(fig_payoff)




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
        V[i,j] = greek_calc(St,klist[j],t,tlist[i],r,q,sigma,optiont)

x, y = np.meshgrid(klist, tlist)
st.title('The Options Sensitivity')
st.write(f"x shape: {x.shape}, y shape: {y.shape}, V shape: {V.shape}")
st.write(f"V values: {V[:5, :5]}")  # Display the first few values of V for inspection

fig = go.Figure(data=[go.Surface(z=V, x=klist, y=tlist)])

# Update layout to enhance visualization
fig.update_layout(
    title='3D Greek Plot',
    scene=dict(
        xaxis_title='Strike (K)',
        yaxis_title='Maturity (T)',
        zaxis_title=f'{greek} Value',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        zaxis=dict(showgrid=True)
    ),
    width=800, height=600
)

# Display the plot in Streamlit
st.plotly_chart(fig)



greek_value = greek_calc(St, K, t, T, r,q, sigma,optiont)
# Display the result
st.write('## Results')
st.write(f'**{greek}**: {round(greek_value,5)}')



    





