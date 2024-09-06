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
st.set_page_config(
    page_title='Stock Options'
)
def d1f(St,K,T,r,q,sigma):
    '''Black scholes merton d1 function'''
    d1 = (math.log(St/K) + (r - q + 0.5 * sigma**2) * (T)) / (sigma * math.sqrt(T))
    return d1

def dN(x):
    '''Probability density function of standard noormal random variable x'''
    return math.exp(-0.5 * x**2) / math.sqrt(2*math.pi)

def N(d):
    '''Cumulative density function of standard normal random variable x'''
    return quad(lambda x: dN(x), -20, d,limit=50)[0]

def BSM_delta(St,K,T,r,q,sigma,optiont):
        """Black-Scholes-Merton Delta of europiean call option"""
        d1 = d1f(St,K,T,r,q,sigma)
        if optiont == "Call":
            delta = N(d1)
        elif optiont == "Put":
            delta = N(d1) - 1
        return delta
def BSM_gamma(St,K,T,r,q,sigma,optiont):
    '''Black Scholes Merton Gamma of a europian option call'''
    d1 = d1f(St,K,T,r,q,sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T))
    return gamma
def BSM_theta(St, K, T, r,q, sigma, optiont):
    '''Black-Scholes-Merton theta of a European option adjusted for dividends'''
    d1 = d1f(St, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    
    if optiont == 'Call':
        theta = -(St * math.exp(-q * (T)) * dN(d1) * sigma / (2 * math.sqrt(T))) \
                - r * K * math.exp(-r * (T)) * N(d2) \
                + q * St * math.exp(-q * (T)) * N(d1)
    elif optiont == 'Put':
        theta = -(St * math.exp(-q * (T)) * dN(d1) * sigma / (2 * math.sqrt(T))) \
                + r * K * math.exp(-r * (T)) * N(-d2) \
                - q * St * math.exp(-q * (T)) * N(-d1)
    
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
def BSM_rho(St,K,T,r,q,sigma,optiont):
    '''Black scholes and merton rho of a europian call ption'''
    d1 = d1f(St,K,T,r,q,sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if optiont == 'Call':
        rho = K * (T) * math.exp(-r * (T)) * N(d2)
    elif optiont == 'Put':
        rho = (N(d2)-1)*K*(T)*math.exp(-r*(T))
    return rho 
def BSM_vega(St,K,T,r,q,sigma,optiont):
    '''Black scholes and merton vega of a europian call option'''
    d1 = d1f(St,K,T,r,q,sigma)
    vega = math.sqrt(T)*St*dN(d1)
    return vega 
def BSM_option_price(St, K, T, r, sigma, option_type='Call', q=0.0):
    """Calculate the Black-Scholes option price for a call or put option considering dividends"""
    d1 = (math.log(St/K) + (r - q + 0.5 * sigma**2) * (T)) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'Call':
        price = St * math.exp(-q * (T)) * N(d1) - K * math.exp(-r * (T)) * N(d2)
    elif option_type == 'Put':
        price = K * math.exp(-r * (T)) * N(-d2) - St * math.exp(-q * (T)) * N(-d1)
    
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

def diff(sigma, St, K, T, r, market_price, option_type):
    '''Calculate the difference between market price and Black-Scholes price'''
    return np.abs(market_price - BSM_option_price(St, K, T, r, sigma, option_type))


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
def new_func(stock):
    stock_price = stock.history(period='1d')['Close'][0]
    return stock_price 
stock_price = new_func(stock)  # Get the latest closing price
St = float(stock_price)
  # Initial guess for volatility

# Fetch available expiration dates for the selected ticker
expirations = stock.options

# Allow the user to select the expiration date
expiration_date = st.sidebar.selectbox("Select Expiration Date", expirations)

# Fetch the option chain for the selected expiration date
option_chain = stock.option_chain(expiration_date)
options = option_chain.calls if optiont == 'Call' else option_chain.puts
# Set the strike price to the closest available strike price (At-The-Money)
strike_prices = options['strike'].tolist()
atm_strike_price = min(strike_prices, key=lambda x: abs(x - St))
K = float(st.sidebar.selectbox('Strike price (K)', strike_prices, index=strike_prices.index(atm_strike_price)))
#strike_prices = options['strike'].tolist()
#K = float(st.sidebar.selectbox('Strike price (K)', strike_prices))
r = st.sidebar.number_input('Risk-free rate (%)', value=4.00, format="%.2f", step=0.10) / 100.0
#r = float(st.sidebar.number_input('Risk-free rate (r)', value=0.040,format="%.3f",step=0.001))
q = (st.sidebar.number_input('Dividend Yield (%)', value=0.00, format="%.2f", step=0.10) / 100.0) + 0.0001
#q = float(st.sidebar.number_input('Dividend Yield (q)', value=0.0, format="%.4f", step=0.001))
sigma = st.sidebar.number_input('Volatility (%)', value=23.73, format="%.2f", step=0.10) / 100.0
#sigma = float(st.sidebar.number_input('Volatility (sigma)', value=0.25,format="%.4f",step=0.001))

initial_sigma = sigma


# Filter for the specific strike price



matching_options = options[(options['strike'] == K)]

if not matching_options.empty:
    T = (pd.to_datetime(expiration_date) - pd.to_datetime('today')).days / 365.0  # Time to maturity in years
    real_market_price = matching_options.iloc[0]['lastPrice']
    market_implied_volatility = matching_options.iloc[0]['impliedVolatility'] * 100  
    st.write(T)
    """"""
    if T <= 0:
        st.error("Time to maturity is zero or has passed. Please select a different expiration date.")
    else:
    # Proceed with the rest of the option price calculation
        option_price = BSM_option_price(St, K, T, r, sigma, optiont)
    # Continue with your logic to display the results
    # Calculate the Black-Scholes option price
    option_price = BSM_option_price(St, K, T, r, sigma, optiont)

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
    difference = option_price - real_market_price
    st.markdown(f"**Difference:** <span style='font-size: 20px;'>${difference:.2f}</span>", unsafe_allow_html=True)
    #st.markdown(f"**Difference:** <span style='font-size: 20px;'>${option_price - real_market_price:.2f}</span></p>", unsafe_allow_html=True)
    st.markdown(f"**Implied Volatility:** <span style='font-size: 20px;'>{market_implied_volatility:.2f}%</span></p>", unsafe_allow_html=True)
    st.markdown(f"**Input Volatility:** <span style='font-size: 20px;'>{sigma*100:.2f}%</span></p>", unsafe_allow_html=True)


if st.sidebar.button("Adjust Volatility to Match Market Price"):
    initial_guess = 1
    result = least_squares(diff, initial_guess, args=(St, K, T, r, real_market_price, optiont))
    adjusted_sigma = result.x[0]
    st.sidebar.write(f"Estimated Implied Volatility: {adjusted_sigma * 100:.3f}%")
    sigma = adjusted_sigma  # Update sigma with the optimized value
    option_price = BSM_option_price(St, K, T, r, sigma, optiont)  # Recalculate option price with optimized sigma

position = st.session_state.position
premium = BSM_option_price(St, K, T, r, sigma, optiont)
strike_min = max(0, K - 60)  # Lower bound: 60 below the selected strike price
strike_max = K + 60          # Upper bound: 60 above the selected strike price
strike_steps = 100   
position = position
# Calculate the option price




# Separate Payoff chart
st.subheader("---------------------------Option Payoff Chart:-------------------------")
st.markdown("""
- The payoff visually demonstrates how the option will perform under different stock price scenarios.
- The chart reflects the chosen parameters, such as position ("Long" or "Short"), option type ("Call" or "Put"), strike price, and other factors.
""")
stock_prices = np.linspace(0.5 * K, 1.5 * K, 100)

payoffs = option_payoff(stock_prices, K,premium, optiont,position)

fig_payoff = plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoffs, label=f'{optiont} Payoff')
plt.xlabel('Stock Price (St)')
plt.ylabel('Payoff')
plt.title(f'{position,optiont} Option Payoff')
plt.grid(True)
plt.legend()
st.pyplot(fig_payoff)


st.subheader("---------------------------Option's Sensitivity--------------------------")

# Add a selection box for user to choose between displaying 3D Greek plot or DataFrame
st.sidebar.header("Choose Display Option")

greek = st.sidebar.selectbox(
    'Select the Greek to plot',
    ('Delta', 'Gamma', 'Theta', 'Vega', 'Rho')
)

# Logic to compute the selected Greek
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

st.subheader('***Greeks Values:***')

# Calculate Greeks for the current values
greek_data = {
    'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
    'Value': [
        BSM_delta(St, K, T, r, q, sigma, optiont),
        BSM_gamma(St, K, T, r, q, sigma, optiont),
        BSM_theta(St, K, T, r, q, sigma, optiont),
        BSM_vega(St, K, T, r, q, sigma, optiont),
        BSM_rho(St, K, T, r, q, sigma, optiont)
    ]
}

# Create DataFrame for Greeks
greek_df = pd.DataFrame(greek_data)
greek_df_T = greek_df.transpose()
# Display the DataFrame
st.dataframe(greek_df_T)

# Now display the 3D Greek Plot

st.subheader('***3D Greek Plot:***')

# Generate plot data
tlist = np.linspace(0.01, 1, 25)
klist = np.linspace(K - 60, K + 60, 50)
V = np.zeros((len(tlist), len(klist)), dtype=np.float16)
for j in range(len(klist)):
    for i in range(len(tlist)):
        V[i, j] = greek_calc(St, klist[j], tlist[i], r, q, sigma, optiont)

fig = go.Figure(data=[go.Surface(z=V, x=klist, y=tlist)])

# Update layout to enhance visualization
fig.update_layout(
    title=f'3D Plot of {greek} Value',
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

# Display the 3D plot
st.plotly_chart(fig)
st.markdown("""
**Note:** The 3D Greek plot is set by default to display **Delta**. You can select a different Greek to plot 
from the dropdown menu in the sidebar. This plot visualizes how the selected Greek changes based on strike price and time to maturity.
""")



# Static Hedge Calculation
st.header("---------------Static Hedge for the Option----------")
st.markdown("""
**Static Hedging** refers to a type of hedging where a position is set up initially and remains unchanged throughout the duration of the hedge. This approach contrasts with **dynamic hedging**, where adjustments to the hedge must be made periodically due to changes in variables like the underlying asset's price, volatility, or time to expiration. In a static hedge, the trader sets the hedge once, and it is maintained without any further adjustments.

According to **John C. Hull's "Options, Futures, and Other Derivatives" (11th edition)**:

> "A hedge is set up initially and never adjusted. Static hedging is sometimes also referred to as 'hedge-and-forget.'"

This makes static hedging simpler and less resource-intensive compared to dynamic hedging. However, it may not be as effective in maintaining a perfectly hedged position, especially when market conditions fluctuate significantly.
""")

# Assume each option contract represents 100 shares
number_of_options = st.sidebar.number_input("Number of Option Contracts", value=1, step=1)
  # Standard contract size, typically 100 shares per contract

# Calculate key parameters for static hedge
delta = BSM_delta(St, K, T, r, q, sigma, optiont)
gamma = BSM_gamma(St, K, T, r, q, sigma, optiont)


# Calculate number of shares to buy or sell to neutralize delta
shares_to_buy_sell = -delta * number_of_options   # Number of shares to hedge
# Determine if the user needs to buy or sell based on position
if st.session_state.position == "Long":
    action = "Sell" if shares_to_buy_sell > 0 else "Buy"
elif st.session_state.position == "Short":
    action = "Buy" if shares_to_buy_sell > 0 else "Sell"



hedge_data = {
    "Greek": ["Delta", "Gamma",  "Action (Buy/Sell)", "Shares to Hedge"],
    "Value": [delta, gamma, action, abs(shares_to_buy_sell)]
}

# Create a pandas DataFrame for the static hedge
hedge_df = pd.DataFrame(hedge_data)
hedge_df_T = hedge_df.transpose()

# Display the static hedge DataFrame
st.subheader("***Static Hedge Table:***")
st.dataframe(hedge_df_T)
st.markdown("""
**Note:** The delta hedge is calculated based on the number of option contracts inputted in the sidebar. 
Ensure that the number of contracts is correctly set to reflect the hedge accurately.
""")# Calculate the number of shares to hedge
shares_to_buy_sell = abs(delta * number_of_options )

# Calculate the cost to hedge
cost_to_hedge = shares_to_buy_sell * St
# Calculate the cost of buying the option
cost_of_option = option_price * number_of_options *100

# Display the cost of buying the option
st.write(f"**Cost of Buying the Option:** ${cost_of_option:.2f}")
# Display the cost to hedge
st.write(f"**Cost to Hedge (Delta Hedge):** ${cost_to_hedge:.2f}")
