import numpy as np
import sys
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf

def Strike_Price(price):
    return round(price/50)*50

def black_scholes(S, K, T, sigma, r, option_type):
    """
    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param sigma: volatility
    :param r: risk-free rate (treasury bills)
    :param option_type: takes string PE(put) or CE(call)
    :return: call price
    """

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
   
    if (option_type == 'CE' or option_type == 'c' or option_type == 'Call' or option_type == 'CALL' or
        option_type == 'call'):
        price= S * N(d1) -  N(d2)* K * np.exp(-r * T)
       
    elif(option_type == 'PE' or option_type == 'p' or option_type == 'Put' or option_type == 'PUT' or
        option_type == 'put'):
        price=  N(-d2)* K * np.exp(-r * T) - S * N(-d1)
    else:
        sys.exit("Option type doesn't match. It should be : \n CE,c,Call,CALL,call for call option \n or \n PE,p,Put,PUT,put for put option.")
       
    return price

def Delta(S, K, r, T, sigma, option_type):
    T=T/(375*252)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = np.exp(-r * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'")
    return delta


def vega(S, K, T, r, sigma):
    """
   
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
   
    """
    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

   
    vega = S  * np.sqrt(T) * N_prime(d1)
    if(round(vega, 5)==0):
        vega=0.00001
   
    return vega

def implied_volatility_call(C, S, K, T, r=0, tol=1.0e-5,
                            max_iterations=1000, tally = False):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity in minute
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :param tally: bool, if true returns sigma in each iterations.
    :return: implied volatility in percent. None if it is negative or Nan.
    '''
   
    if T==0 :
        T=0.1
       
    T=T/(252*375)
    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    if tally == True :

        for i in range(max_iterations):
           

           
            ### calculate difference between blackscholes price and market price with
            ### iteratively updated volality estimate
            diff = black_scholes(S, K, T, sigma, r, option_type='call') - C

            ###break if difference is less than specified tolerance level
            if abs(diff) < tol:
                print(f'found on {i}th iteration')
                print(f'difference is equal to {diff}')
                break

            ### use newton rapshon to update the estimate
            sigma = sigma - diff / vega(S, K, T, r, sigma)
            if sigma < 0:
                sigma = None
                break
           
        # if sigma==None:
        #     print(f'sigma is less than 0 or Nan. Default is considered.')
        # print(f"sigma is determined as {sigma}")
        # print("===========End of Iterations================\n\n")
       
    else:
       
        for i in range(max_iterations):
           
            ### calculate difference between blackscholes price and market price with
            ### iteratively updated volality estimate
            diff = black_scholes(S, K, T, sigma, r, option_type='call') - C

            ###break if difference is less than specified tolerance level
            if abs(diff) < tol:
                break

            ### use newton rapshon to update the estimate
            sigma = sigma - diff / vega(S, K, T, r, sigma)
            if sigma <= 0 or sigma == None:
               
                # print(f"sigma is determined as None.")
                sigma = None
                break
   
    return sigma



def implied_volatility_put(C, S, K, T, r=0, tol=1.0e-5,
                            max_iterations=1000, tally = False):
    '''

    :param C: Observed put price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity in minute
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :param tally: bool, if true returns sigma in each iterations.
    :return: implied volatility in percent. None if it is negative or Nan.
    '''
   
    if T==0 :
        T=0.1
       
    T=T/(375*252)
    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    if tally == True :

        for i in range(max_iterations):
           

           
            ### calculate difference between blackscholes price and market price with
            ### iteratively updated volality estimate
            diff = black_scholes(S, K, T, sigma, r, option_type='put') - C

            ###break if difference is less than specified tolerance level
            if abs(diff) < tol:
                print(f'found on {i}th iteration')
                print(f'difference is equal to {diff}')
                break

            ### use newton rapshon to update the estimate
            sigma = sigma - diff / vega(S, K, T, r, sigma)
            if sigma < 0:
                sigma = None
                break
           
        if sigma==None:
            print(f'sigma is less than 0 or Nan. Default is considered.')
        print(f"sigma is determined as {sigma}")
        print("===========End of Iterations================\n\n")
       
    else:
       
        for i in range(max_iterations):
           
            ### calculate difference between blackscholes price and market price with
            ### iteratively updated volality estimate
            diff = black_scholes(S, K, T, sigma, r, option_type='put') - C

            ###break if difference is less than specified tolerance level
            if abs(diff) < tol:
                break

            ### use newton rapshon to update the estimate
            sigma = sigma - diff / vega(S, K, T, r, sigma)
            if sigma <= 0 or sigma == None:
               
#                 print(f"sigma is determined as None.")
                sigma = None
                break
   
    return sigma

