import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override() 
from bs4 import BeautifulSoup
import requests
import sys
import os

# ######################### #
# DISABLE / ENABLE PRINTING #
# ######################### #

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# ######################### #
# GET FTSE100 STOCK TICKERS #
# ######################### #

def get_ftse100_stock_tickers():
    """
    Gathers tickers for all stocks in the FTSE100. The list of stocks is web-scraped
    from the Hargreaves Lansdown website.

    Returns:
       list: List of tickers for all stocks in the FTSE100.
    """

    url = "https://www.hl.co.uk/shares/stock-market-summary/ftse-100"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the table containing the stock data
    table = soup.find("table", {"class": "stockTable"})
    
    stocks = []
    
    if table is not None:
        # Find all the rows in the table
        rows = table.find_all("tr")
        
        # Skip the header row
        for row in rows[1:]:
            # Extract the stock symbol from the first column
            symbol = row.find("td").text.strip()
            stocks.append(symbol)
    
    return stocks

# ################ #
# GET STOCK PRICES #
# ################ #

def get_stock_prices(ticker, start="2018-01-01", end="2023-01-01"):
    """
    Gathers pricing data for the provided stock across a given range of dates.

    Parameters:
        ticker: The ticker for the stock who's price data is being gathered.
        start: The start date for the pricing data.
        end: The end date for the pricing data.

    Returns:
        ndarray: List of closing prices for the stock across the given date range.
    """

    return pdr.get_data_yahoo(ticker, start=start, end=end, progress=False)["Close"].to_numpy()

# ###################### #
# ANALYSE FTSE100 STOCKS #
# ###################### #

def analyse_ftse100_stocks():
    """
    Performs an analysis of all of the stocks in the FTSE100.

    - The stock data for all stocks in the FTSE100 over a specified time period is downloaded via web-scraping and 
      Yahoo Finance.
    - This data is filtered to a list of closing prices for the stocks.
    - This data is filtered to contain only stocks which have data over the entire time period. 
        - This simple method is used to make it possible to compute the correlation coefficient easily. 
        - A more complex strategy could be used to include those stocks for which only a few datapoints are missing.
    - A correlation matrix is computed using the Pearson Correlation Coefficient between each of the stocks in the list.
    - The most highly correlated stocks are extracted based on this correlation matrix, and:
        - Their price ratio is calculated.
        - Their price spread is calculated.
        - Differencing is applied until the ratio/spread of the pair of stocks until the ADF test is passed.
    - This information is then outputted to the user.
    - Plots are then made of:
        - The correlation matrix for all of the filtered stocks.
        - The relative price change between the two most correlated stocks.
        - The Z-score for the ratio in price between the most correlated stocks.
        - The Z-score for the spread in price between the most correlated stocks.
    - Ultimately, the function computes the pair of stocks that are most suitable for pairs trading, and presents 
      information to justify this decision in the form of statistics and figures.
    """

    print("## ########################################## ##")
    print("## ANALYSING FTSE100 STOCKS FOR PAIRS TRADING ##")
    print("## ########################################## ##")
    print()
    
    # gathering stock tickers
    ftse100_stock_tickers = get_ftse100_stock_tickers()
    
    # defining time range
    start = "2018-01-01"
    end = "2023-01-01"
    
    
    # gathering the prices for the stocks
    print("Downloaded stock pricing information...")
    #disablePrint()
    stock_prices = []
    stock_tickers = [] 
    for ticker in ftse100_stock_tickers:
        prices = get_stock_prices(ticker, start, end)
        if len(prices) != 0:
            stock_prices.append(prices.tolist())
            stock_tickers.append(ticker)
    #enablePrint()
    print("Stock pricing information downloaded.")
    print()
           
    
    # filtering to only get stocks which have the same amount of price data
    print("Filtering stock prices...")
    lengths = [len(stock) for stock in stock_prices]

    final_stock_prices = []
    final_stock_tickers = []

    for i in range(len(stock_prices)):
        stock = stock_prices[i]
        ticker = stock_tickers[i]
        if len(stock) == max(lengths):
            final_stock_prices.append(stock)
            final_stock_tickers.append(ticker)
    print("Stock prices filtered.")
    print("Total number of stocks : ", len(final_stock_prices))
    print()
    

    # calculating the correlation matrix for the stocks
    print("Determining stock price correlation...")
    correlation_matrix = np.corrcoef(final_stock_prices)
    print("Stock price correlation determined...")
    print()
    
    print("Determining most correlated stocks...")
    # Find indices of maximum correlation values (excluding the main diagonal)
    indices = np.argwhere(correlation_matrix > 0.9)

    # removing diagonals (correlation between stock and itself)
    final_indices = []
    for index in indices:
        if index[0] != index[1]:
            final_indices.append(index)

    # making list of most correlated stocks and sorting based on correlation
    most_correlated_stocks = [(i, j, correlation_matrix[i, j]) for i, j in final_indices]
    most_correlated_stocks.sort(key=lambda x: x[2], reverse=True)

    # outputting most correlated stocks
    i, j, correlation = most_correlated_stocks[0]
    print(f"Most + correlated stocks determined : {final_stock_tickers[i]} and {final_stock_tickers[j]} with {correlation}")
    print()
    
    # analysing ratio and spread for pair
    print(f"Analysing ratio and sprad for pair {final_stock_tickers[i]} and {final_stock_tickers[j]}...")
    
    ratio = np.divide(final_stock_prices[i], final_stock_prices[j])
    ratio_z_score = (ratio - np.mean(ratio)) / np.std(ratio)
    
    diff = np.subtract(final_stock_prices[i], final_stock_prices[j])
    diff_z_score = (diff - np.mean(diff)) / np.std(diff)
    print(f"Ratio and spread for pair {final_stock_tickers[i]} and {final_stock_tickers[j]} analysed.")
    print()
    
    # figures for plots
    print("Plotting figures...")
    fig, axes = plt.subplots(4, 1, figsize=(6,18))
    fig.tight_layout(pad=5.0)
    
    # plotting the correlation matrix
    image = axes[0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(image, ax=axes[0], label='Pearson Correlation')
    axes[0].set_title('FTSE100 Stock Correlation Matrix')
    axes[0].set_xticks(np.arange(len(final_stock_prices)))
    axes[0].set_xticklabels(final_stock_tickers, rotation=90)
    axes[0].set_yticks(np.arange(len(final_stock_prices)))
    axes[0].set_yticklabels(final_stock_tickers)
    axes[0].tick_params(axis='both', which='both', length=0)  # Remove tick marks
    axes[0].set_aspect('equal')  # Set aspect ratio to equal
    
    # plotting relative price change between stocks
    axes[1].plot(np.divide(final_stock_prices[i], final_stock_prices[i][0]), label=final_stock_tickers[i])
    axes[1].plot(np.divide(final_stock_prices[j], final_stock_prices[j][0]), label=final_stock_tickers[j])
    axes[1].set_title(f"Relative Price Change in {final_stock_tickers[i]} and {final_stock_tickers[j]} \n (PCC = %1.2f)"%correlation)
    axes[1].set_xlabel("Time")
    axes[1].set_xticks([])
    axes[1].set_ylabel("Relative Price Change")
    axes[1].legend()
    
    # plotting z-score for price ratio between stocks
    axes[2].plot(ratio_z_score)
    axes[2].axhline(y=0, color='k')
    axes[2].axhline(y=1, color='r')
    axes[2].axhline(y=-1, color='r')
    axes[2].set_title(f"Z-Score for Ratio in Price Between {final_stock_tickers[i]} and {final_stock_tickers[j]} \n ({final_stock_tickers[i]} / {final_stock_tickers[j]})")
    axes[2].set_xlabel("Time")
    axes[2].set_xticks([])
    axes[2].set_ylabel("Z Score of Price Ratio")
    
    # plotting difference (spread) between the stocks
    axes[3].plot(diff_z_score)
    axes[3].axhline(y=0, color='k')
    axes[3].axhline(y=1, color='r')
    axes[3].axhline(y=-1, color='r')
    axes[3].set_title(f"Z-Score for Spread in Price Between {final_stock_tickers[i]} and {final_stock_tickers[j]} \n ({final_stock_tickers[i]} - {final_stock_tickers[j]})")
    axes[3].set_xlabel("Time")
    axes[3].set_xticks([])
    axes[3].set_ylabel("Z Score of Spread")
    
    print("Figures plotted.")

# ################## #
# TRADING STRATEGY 1 #
# ################## #

def deviation_from_mean(stock_1, stock_2):
    """
    Performs back-testing of a deviation from the mean pairs trading strategy on
    the provided stocks.

    The strategy works as follows:
        - Maintain long possitions in both stocks by default.
        - Monitor the z-score for the ratio between the two stocks at each timestep.
        - If the z-score goes above a threshold, enter opposing long and short 
          positions on the stocks.
        - If the z-score returns to the aceptable range, revert back to long
          positions on both stocks.

    Parameters:
        stock_1: ticker, prior knowledge and future values for the first stock.
        stock_2: ticker, prior knowledge and future values for the second stock.

    Returns:
        list: List of values representing the lengths of each of the trades the 
              strategy enters.
        list: List of values representing the profits the strategy makes off of
              each trade.
        list: Record of the amount of money the strategy has on each day.
    """

    # extracting data
    stock_1_ticker = stock_1[0]
    stock_1_prev = stock_1[1]
    stock_1_future = stock_1[2]
    stock_1 = {stock_1_ticker: [stock_1_prev, stock_1_future]}
    
    stock_2_ticker = stock_2[0]
    stock_2_prev = stock_2[1]
    stock_2_future = stock_2[2]
    stock_2 = {stock_2_ticker: [stock_2_prev, stock_2_future]}
    
    # defining initial parameters
    money = 200 # 100 per stock initially
    stock_1_current = stock_1_prev
    stock_2_current = stock_2_prev
    is_in_trade = False
    baseline_long_position_stock_1 = [0, 0, 0] # entry price, amount, timestep
    baseline_long_position_stock_2 = [0, 0, 0] # entry price, amount, timestep
    current_long_position = ['TICKER', 0, 0, 0] # stock in long, entry price, amount, timestep
    current_short_position = ['TICKER', 0, 0, 0] # stock in short, entry price, amount, timestep
    threshold = 1
    
    # records for tracking
    daily_money = []
    number_of_trades = 0
    length_of_trades = []
    trade_profits = []
    
    # entering long position on both stocks
    baseline_long_position_stock_1 = [stock_1_future[0], money/2, 0]
    baseline_long_position_stock_2 = [stock_2_future[0], money/2, 0]
    
    # iterating through time series future
    for i in range(len(stock_1_future)):
        
        # updating current stock info
        stock_1_current = np.append(stock_1_current, stock_1_future[i])
        stock_2_current = np.append(stock_2_current, stock_2_future[i])
        
        # defining ratio
        ratio = np.divide(stock_1_current, stock_2_current)
        ratio_z_score = (ratio - np.mean(ratio)) / np.std(ratio)
        
        # checking if in trade
        if is_in_trade:
            # in trade -> need to check for exit criteria
            
            # calculating current amount of money
            if current_long_position[0] == stock_1_ticker:
                long_position_change = (stock_1_current[-1] - current_long_position[1]) / current_long_position[1]
                short_position_change = (current_short_position[1] - stock_2_current[-1]) / current_short_position[1]
            else:
                long_position_change = (stock_2_current[-1] - current_long_position[1]) / current_long_position[1]
                short_position_change = (current_short_position[1] - stock_1_current[-1]) / current_short_position[1]

            long_money = current_long_position[2] * (1 + long_position_change)
            short_money = current_short_position[2] * (1 + short_position_change)
            current_money = long_money + short_money
            daily_money.append(current_money)
                
            # checking exit criteria
            if ratio_z_score[-1] <=threshold and ratio_z_score[-1] >= -threshold:
                # exit criteria met -> need to exit trade
                #print(f"Need to exit trade on day {i}. z-score in valid range.")
                
                # exiting trade
                is_in_trade = False
                
                profit = current_money - money
                money = current_money
                #print(f"\tProfit from trade : {profit}")
                #print(f"\tNew Money : {money}")
                
                # updating records
                length_of_trades.append(i - current_long_position[3])
                trade_profits.append((long_position_change + short_position_change) * 100)
                
                # entering long position on both stocks
                baseline_long_position_stock_1 = [stock_1_current[-1], money/2, 0]
                baseline_long_position_stock_2 = [stock_2_current[-1], money/2, 0]
                
                
        else:
            # not in trade -> need to check for entry criteria
            
            # updating record
            stock_1_change = (stock_1_current[-1] - baseline_long_position_stock_1[0]) / baseline_long_position_stock_1[0]
            stock_2_change = (stock_2_current[-1] - baseline_long_position_stock_2[0]) / baseline_long_position_stock_2[0]
            stock_1_money = baseline_long_position_stock_1[1] * (1 + stock_1_change)
            stock_2_money = baseline_long_position_stock_2[1] * (1 + stock_2_change)
            money = stock_1_money + stock_2_money
            daily_money.append(money)
        
            # checking entry criteria 1
            if ratio_z_score[-1] >= threshold:
                # stock 1 overvalued, stock 2 undervalued
                # short stock 1, long stock 2

                #print(f"Need to enter trade on day {i}. z-score >= 1.")
                #print("\tGoing short on stock 1 and long on stock 2")
                
                is_in_trade = True
                number_of_trades += 1

                # going short on stock 1
                current_short_position = [stock_1_ticker, stock_1_future[i], money / 2, i]

                # going long on stock 2
                current_long_position = [stock_2_ticker, stock_2_future[i], money / 2, i]

            # checking entry criteria 2
            elif ratio_z_score[-1] <= -threshold:
                # stock 1 undervalued, stock 2 overvalued
                # long stock 1, short stock 2

                # print(f"Need to enter trade on day {i}. z-score <= -1")
                # print("\tGoing long on stock 1 and short on stock 2")

                # entering trade
                is_in_trade = True
                number_of_trades += 1

                # going long on stock 1
                current_long_position = [stock_1_ticker, stock_1_future[i], money / 2, i]

                # going short on stock 2
                current_short_position = [stock_2_ticker, stock_2_future[i], money / 2, i]
    
    # returrning records 
    return length_of_trades, trade_profits, daily_money

# ################## #
# TRADING STRATEGY 2 #
# ################## #

def compute_moving_average(time_series, n):
    """
    Computes the n-timestep moving average for a time series.

    Parameters:
        time_series: The time series the n-timestep moviing average is being
        calculated for.
        n: The number of time steps the moving average is dependent on.

    Returns:
        float: The n-timestep moving average for the time series.
    """

    total = 0
    for i in range (n):
        total += time_series[-1-i]
    average = total / n
    return average

def time_series_momentum(stock_1, stock_2):
    """
    Performs back-testing of a time series momentum pairs trading strategy on
    the provided stocks.

    The strategy works as follows:
        - Maintain long possitions in both stocks by default.
        - Monitor the SMA and LMA for the z-score of the ratio between the two stocks 
          at each timestep.
        - If the differene goes above a threshold, enter opposing long and short 
          positions on the stocks.
        - If the difference returns to the aceptable range, revert back to long
          positions on both stocks.

    Parameters:
        stock_1: ticker, prior knowledge and future values for the first stock.
        stock_2: ticker, prior knowledge and future values for the second stock.

    Returns:
        list: List of values representing the lengths of each of the trades the 
              strategy enters.
        list: List of values representing the profits the strategy makes off of
              each trade.
        list: Record of the amount of money the strategy has on each day.
    """
    
    # extracting data
    stock_1_ticker = stock_1[0]
    stock_1_prev = stock_1[1]
    stock_1_future = stock_1[2]
    stock_1 = {stock_1_ticker: [stock_1_prev, stock_1_future]}
    
    stock_2_ticker = stock_2[0]
    stock_2_prev = stock_2[1]
    stock_2_future = stock_2[2]
    stock_2 = {stock_2_ticker: [stock_2_prev, stock_2_future]}
    
    # defining initial parameters
    money = 200 # 100 per stock initially
    stock_1_current = stock_1_prev
    stock_2_current = stock_2_prev
    is_in_trade = False
    baseline_long_position_stock_1 = [0, 0, 0] # entry price, amount, timestep
    baseline_long_position_stock_2 = [0, 0, 0] # entry price, amount, timestep
    current_long_position = ['TICKER', 0, 0, 0] # stock in long, entry price, amount, timestep
    current_short_position = ['TICKER', 0, 0, 0] # stock in short, entry price, amount, timestep
    threshold = 0.65
    
    # entering long position on both stocks
    baseline_long_position_stock_1 = [stock_1_future[0], money/2, 0]
    baseline_long_position_stock_2 = [stock_2_future[0], money/2, 0]
    
    # records for tracking
    daily_money = []
    number_of_trades = 0
    length_of_trades = []
    trade_profits = []
    
    # iterating through time series future
    for i in range(len(stock_1_future)):
        
        # updating current stock info
        stock_1_current = np.append(stock_1_current, stock_1_future[i])
        stock_2_current = np.append(stock_2_current, stock_2_future[i])
        
        # defining ratio
        ratio = np.divide(stock_1_current, stock_2_current)
        
        # computing SMA and LMA for ratio
        ratio_sma = compute_moving_average(ratio, 20)
        ratio_lma = compute_moving_average(ratio, 200)
        
        current_diff = abs(ratio_lma - ratio_sma)
        
        # sma >= lma + threshold 
        
        # checking if in trade
        if is_in_trade:
            # in trade -> need to check for exit criteria
            
            # calculating current amount of money
            if current_long_position[0] == stock_1_ticker:
                long_position_change = (stock_1_current[-1] - current_long_position[1]) / current_long_position[1]
                short_position_change = (current_short_position[1] - stock_2_current[-1]) / current_short_position[1]
            else:
                long_position_change = (stock_2_current[-1] - current_long_position[1]) / current_long_position[1]
                short_position_change = (current_short_position[1] - stock_1_current[-1]) / current_short_position[1]

            # updating record
            long_money = current_long_position[2] * (1 + long_position_change)
            short_money = current_short_position[2] * (1 + short_position_change)
            current_money = long_money + short_money
            daily_money.append(current_money)
            
            # checking exit criteria
            if current_diff < threshold:
                # exit criteria met -> need to exit trade
                #print(f"Need to exit trade on day {i}. SMA-LMA in valid range.")
                
                # exiting trade
                is_in_trade = False
                
                profit = current_money - money
                money = current_money
                #print(f"\tProfit from trade : {profit}")
                #print(f"\tNew Money : {money}")
                
                # updating records
                length_of_trades.append(i - current_long_position[3])
                trade_profits.append((long_position_change + short_position_change) * 100)
                
                # entering long position on both stocks
                baseline_long_position_stock_1 = [stock_1_current[-1], money/2, 0]
                baseline_long_position_stock_2 = [stock_2_current[-1], money/2, 0]
        
        else:
            # not in trade -> need to check for entry criteria
            
            # updating record
            stock_1_change = (stock_1_current[-1] - baseline_long_position_stock_1[0]) / baseline_long_position_stock_1[0]
            stock_2_change = (stock_2_current[-1] - baseline_long_position_stock_2[0]) / baseline_long_position_stock_2[0]
            stock_1_money = baseline_long_position_stock_1[1] * (1 + stock_1_change)
            stock_2_money = baseline_long_position_stock_2[1] * (1 + stock_2_change)
            money = stock_1_money + stock_2_money
            daily_money.append(money)
        
            # checking entry criteria 1
            if (ratio_sma < ratio_lma + threshold):
                # trend = stock 1 going up, stock 2 going down
                # long stock 1, short stock 2

                #print(f"Need to enter trade on {i}.")
                #print("\tGoing long on stock 1 and short on stock 2")
                
                is_in_trade = True
                number_of_trades += 1

                # going long on stock 1
                #print(f"Current Long position = [{stock_1_ticker}, {stock_1_future[i]}, {money / 2}]")
                current_long_position = [stock_1_ticker, stock_1_future[i], money / 2, i]

                # going short on stock 2
                #print(f"Current Short position = [{stock_2_ticker}, {stock_2_future[i]}, {money / 2}]")
                current_short_position = [stock_2_ticker, stock_2_future[i], money / 2, i]

            # checking entry criteria 1
            if (ratio_sma > ratio_lma - threshold):
                # trend = stock 1 going down, stock 2 going up
                # short stock 1, long stock 2

                #print(f"Need to enter trade on {i}.")
                #print("\tGoing short on stock 1 and long on stock 2")
                
                is_in_trade = True
                number_of_trades += 1

                # going short on stock 1
                #print(f"Current Long position = [{stock_1_ticker}, {stock_1_future[i]}, {money / 2}]")
                current_short_position = [stock_1_ticker, stock_1_future[i], money / 2, i]

                # going long on stock 2
                #print(f"Current Short position = [{stock_2_ticker}, {stock_2_future[i]}, {money / 2}]")
                current_long_position = [stock_2_ticker, stock_2_future[i], money / 2, i]
    
    # returrning records 
    return length_of_trades, trade_profits, daily_money

# ###### #
# PART 2 #
# ###### #

def part_2():
    # defining time range
    start = "2018-01-01"
    end = "2023-01-01"
    
    # defining stocks to be used in backtesting
    stock_1_ticker = "TSCO"
    stock_1_price = get_stock_prices(stock_1_ticker, start, end)
    stock_1_prev = stock_1_price[:round(len(stock_1_price) / 2)]
    stock_1_future = stock_1_price[round(len(stock_1_price) / 2):]
    stock_1 = [stock_1_ticker, stock_1_prev, stock_1_future]

    stock_2_ticker = "PSH"
    stock_2_price = get_stock_prices(stock_2_ticker, start, end)
    stock_2_prev = stock_2_price[:round(len(stock_2_price) / 2)]
    stock_2_future = stock_2_price[round(len(stock_2_price) / 2):]
    stock_2 = [stock_2_ticker, stock_2_prev, stock_2_future]
    
    # calculating baseline profit for each stock
    baseline_profit = [0]
    for i in range (1, len(stock_1_future)):
        stock_1_profit = (stock_1_future[i] - stock_1_future[0]) / stock_1_future[0] * 100
        stock_2_profit = (stock_2_future[i] - stock_2_future[0]) / stock_2_future[0] * 100
        baseline_profit.append((stock_1_profit + stock_2_profit) / 2)
    
    # backtesting the stocks on the two strategies
    dfm_length_of_trades, dfm_trade_profits, dfm_daily_money = deviation_from_mean(stock_1, stock_2)
    tsm_length_of_trades, tsm_trade_profits, tsm_daily_money = time_series_momentum(stock_1, stock_2)

    
    # calculating total profit for each strat
    dfm_total_profit = dfm_daily_money[-1] - dfm_daily_money[0]
    tsm_total_profit = tsm_daily_money[-1] - tsm_daily_money[0]
    
    # printing baselines
    print("Baselines:")
    stock_1_profit = (stock_1_future[-1] - stock_1_future[0]) / stock_1_future[0] * 100
    stock_2_profit = (stock_2_future[-1] - stock_2_future[0]) / stock_2_future[0] * 100
    print(f"\tStock 1 Baseline Profit : {stock_1_profit} %")
    print(f"\tStock 2 Baseline Profit : {stock_2_profit} %")
    print(f"\tAverage Baseline Profit : {(stock_1_profit + stock_2_profit) / 2} %")
    
    # printing results
    print("Diversion From Mean Statistics:")
    print(f"\tTotal Profit : {dfm_total_profit} ({dfm_total_profit / dfm_daily_money[0] * 100}%)")
    print(f"\tTotal number of trades : {len(dfm_length_of_trades)}")
    print(f"\tMean length of trade : {np.mean(dfm_length_of_trades)}")
    print(f"\tTotal time spent in trade : {np.sum(dfm_length_of_trades)} ({np.sum(dfm_length_of_trades) / len(stock_2_future)} %)")
    print(f"\tMean trade Profit (%) : {np.mean(dfm_trade_profits)}")
    
    print("Time Series Momentum Statistics:")
    print(f"\tTotal Profit : {tsm_total_profit} ({tsm_total_profit / tsm_daily_money[0] * 100}%)")
    print(f"\tTotal number of trades : {len(tsm_length_of_trades)}")
    print(f"\tMean Length of trade : {np.mean(tsm_length_of_trades)}")
    print(f"\tTotal time spent in trade : {np.sum(tsm_length_of_trades)} ({np.sum(tsm_length_of_trades) / len(stock_2_future)} %)")
    print(f"\tMean Trade Profit (%) : {np.mean(tsm_trade_profits)}")
    
    # plotting relative change in money
    dfm_profit = np.divide(dfm_daily_money, dfm_daily_money[0])
    dfm_profit = (dfm_profit - 1) * 100
    tsm_profit = np.divide(tsm_daily_money, tsm_daily_money[0])
    tsm_profit = (tsm_profit - 1) * 100
    
    fig, axes = plt.subplots(1, 1, figsize=(6,5))
    fig.tight_layout(pad=5.0)
    
    axes.plot(dfm_profit, label='Strategy 1')
    axes.plot(tsm_profit, label='Strategy 2')
    axes.plot(baseline_profit, label='Baseline')
    axes.set_title(f"% Profit of Trading Strategies")
    axes.set_xlabel("Time")
    axes.set_xticks([])
    axes.set_ylabel("% Profit")
    axes.legend()

# #### #
# MAIN #
# #### #

if __name__ == "__main__":
    # please un-comment the part you would like to run
    
    # analyse_ftse100_stocks()

    part_2()