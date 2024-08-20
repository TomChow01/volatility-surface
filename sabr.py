import numpy as np
import pandas as pd
from scipy.optimize import minimize

def sabr_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        term1 = alpha / (F**(1 - beta))
        term2 = 1 + ( ( (1 - beta)**2 ) / 24 ) * ( alpha**2 / (F**(2 - 2 * beta)) ) * T
        term3 = (rho * beta * nu * alpha) / (4 * F**(1 - beta)) * T
        term4 = ( (2 - 3 * rho**2) / 24 ) * (nu**2) * T
        return term1 * (term2 + term3 + term4)
    else:
        logFK = np.log(F / K)
        FK_beta = (F * K)**((1 - beta) / 2)
        z = (nu / alpha) * FK_beta * logFK
        xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        term1 = alpha / (FK_beta * (1 + (1 - beta)**2 / 24 * logFK**2 + (1 - beta)**4 / 1920 * logFK**4))
        term2 = (1 + ((1 - beta)**2 / 24 * logFK**2 + (1 - beta)**4 / 1920 * logFK**4)) * T
        return term1 * (1 + (term2) * ((z / xz) - 1))
    
def sabr_calibration(F, strikes, T, market_vols, beta):
    def loss_function(params):
        alpha, rho, nu = params
        model_vols = [sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        return np.sum((np.array(model_vols) - np.array(market_vols))**2)
    
    # Initial guess for alpha, rho, nu
    initial_guess = [0.2, 0.0, 0.2]
    bounds = [(0, None), (-0.999, 0.999), (0, None)]
    
    result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x  # Return the calibrated alpha, rho, nu


def calculate_implied_volatility(option_chain, F, beta=0.5):
    unique_expiries = option_chain['Expiry'].unique()
    implied_volatilities = []

    for expiry in unique_expiries:
        expiry_data = option_chain[option_chain['Expiry'] == expiry]
        print('expiry_data:', expiry_data)
        strikes = expiry_data['Strike price'].values
        print('strikes:', strikes)
        market_vols = expiry_data['Close'].values
        T = expiry_data['Time'].values[0]
        
        # Calibrate SABR model
        alpha, rho, nu = sabr_calibration(F, strikes, T, market_vols, beta)
        
        # Calculate implied volatilities using the calibrated SABR model
        sabr_vols = [sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        print('sabr vols:', sabr_vols)
        expiry_data['Implied Volatility'] = sabr_vols
        implied_volatilities.append(expiry_data)
    
    return pd.concat(implied_volatilities)


option_chain = {
    'Expiry': ['2022-06-30', '2022-06-30', '2022-6-30'],
    'Strike price': [90, 100, 110],
    'Time': [0.4, 0.5, 0.6],
    'Close': [0.1, 0.2, 0.3]}

option_chain = pd.DataFrame(option_chain)

if __name__ == "__main__":
    
    F = 100
    K = 100
    T = 1
    alpha = 0.2
    beta = 0.5
    rho = 0.2
    nu = 0.5
    # 
    print(sabr_vol(F, K, T, alpha, beta, rho, nu))
    print(calculate_implied_volatility(option_chain, F, beta))