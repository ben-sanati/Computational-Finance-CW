import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override() 

# ################ #
# GET STOCK PRICES #
# ################ #

def get_stock_prices(ticker):
    """
    Gathers pricing data for the provided stock across a fixed range of dates.

    Parameters:
        ticker: The ticker for the stock who's price data is being gathered.

    Returns:
        ndarray: List of closing prices for the stock across a fixed date range.
    """
    
    return pdr.get_data_yahoo(ticker, start="2022-01-01", end="2022-12-31")["Close"].to_numpy()

# ############ #
# COMPUTE MAPE #
# ############ #

def calculate_mape(y_true, y_pred):
    """
    Computes the mean absolute percentage error (MAPE) between two NumPy arrays.

    Parameters:
        y_true (ndarray): Array containing the true values.
        y_pred (ndarray): Array containing the predicted values.

    Returns:
        float: Mean absolute percentage error (MAPE) between y_true and y_pred.
    """
    
    # Ensure that the input arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Calculate the absolute percentage error
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)

    # Handle cases where the true value is zero to avoid division by zero
    absolute_percentage_error = np.nan_to_num(absolute_percentage_error, nan=0.0, posinf=0.0)

    # Calculate the mean absolute percentage error (MAPE)
    mape = np.mean(absolute_percentage_error) * 100

    return mape

# ############# #
# COMPUTE ARIMA #
# ############# #

def compute_arima(time_series, p, d, q, steps_ahead):
    """
    Computes the ARIMA model for the given time series based on the specified parameters.

    Parameters:
        time_series: The input time series data.
        p: List of weights for the AR component.
        d: The order of differencing.
        q: List of weights for the MA component
        steps_ahead: The number of steps to forecast ahead.

    Returns: 
        An array containing the predicted values for the time series.
    """

    # differences
    diffs = []
    for i in range (1, d+1):
        diff_i = np.diff(time_series, i)
        for j in range (i):
            diff_i = np.insert(diff_i, 0, np.nan)
        diffs.append(diff_i)
    diff = diffs[-1]
    
    # difference predictions
    diffs_pred = []
    for i in range (d):
        diffs_pred.append(np.array([]))

    # errors in differrence predictions
    error = np.array([])

    # actual predictions
    pred = np.array([])

    # iterating through time series to get predictions
    for t in range (0, len(time_series) + steps_ahead):
        
        # can only make prediction if have enough data for AR term
        if(t <= len(diff)):
            if t <= len(p) or np.isnan(diff[t-len(p)]):
                for i in range(d):
                    diffs_pred[i] = np.append(diffs_pred[i], np.nan)
                error = np.append(error, np.nan)
                pred = np.append(pred, np.nan)
                continue

        # AR component
        ar = 0
        for i in range (0, len(p)):
            # can only use real diff it has not run out
            if(t <= len(diff)): # <= because use t-i for t, so can use index t
                ar += p[i] * diff[t-i-1]
            else:
                # have to use diff predictions if actual diff run out
                ar += p[i] * diffs_pred[-1][t-i-1]

        # MA component
        ma = 0
        for i in range (len(q)):           
            #  checking error value is there                  
            if not(np.isnan(error[t-i-1])):
                ma += q[i] *  error[t-i-1]

        # diff pred
        diff_pred_t = ar + ma
        diffs_pred[-1] = np.append(diffs_pred[-1], diff_pred_t)

        # error (if possible with actual diff array)
        if t < len(diff):
            error_t = diff[t] - diff_pred_t
            error = np.append(error, error_t)
        else:
            # error not possible -> just add previous error
            error = np.append(error, np.mean(error))
            
        # undifferencing
        for i in range (d - 1):
            if(t <= len(time_series)):
                diff_pred_t = diff_pred_t + diffs[-i-2][-2]
            else:
                diff_pred_t = diff_pred_t + diffs_pred[-2-i][-2]
            diffs_pred[-2-i] = np.append(diffs_pred[-2-i], diff_pred_t)
        
        if(t <= len(time_series)): # <= bcus can use t-1 for t, so can use index t
            pred_t = time_series[t-1] + diff_pred_t
            pred = np.append(pred, pred_t)
        else:
            pred_t = pred[t-1] + diff_pred_t
            pred = np.append(pred, pred_t)
    
    # returning predictions
    return pred

# ########### #
# TRAIN ARIMA #
# ########### #

def train_arima(training_time_series, p_range, d_range, q_range, learning_rate, epochs):
    """
    Fits an ARIMA model to the provided training time series. The method performs gradient
    descent across all possible arima models based on the provided range of model parameters
    to find the optimal fit.

    Parameters:
        training_time_series: The time series the ARIMA model will be fit to.
        p_range: A list containing the range of p values to consider.
        d_range: A list containing the range of d values to consider.
        q_range: A list containing the range of q values to consider.
        learning_rate: The learning rate for the gradient descent algorithm.
        epochs: The number of epochs to perform gradient descent for each ARIMIA 
                configuration.

    Returns:
        list: Optimal weights for AR component.
        int: Optimal value for d.
        list: Optimal weights for MA component.
        list: History of MAPE during the training process.
    """

    # mape values
    mapes = []
    
    # current best loss
    best_mape = 100000
    
    # best p, d and q
    best_p = []
    best_d = 1
    best_q = []
    
    # list of all possible combinations
    ps, ds, qs = np.meshgrid(p_range, d_range, q_range, indexing='ij')
    combinations = np.vstack([ps.ravel(), ds.ravel(), qs.ravel()]).T
    
    for combination in (combinations):
        p_len = combination[0]
        p = p = np.random.uniform(-1 / p_len, 1 / p_len, size=p_len)
        d = combination[1]
        q_len = combination[2]
        q = q = np.random.uniform(-1 / q_len, 1 / q_len, size=q_len)
        
        print("Searching space of : ARIMA(",p_len,",",d,",",q_len,")...")
    
        # Iterate for the specified number of iterations
        for epoch in range(epochs):
            # Compute predictions using current weights
            pred = compute_arima(training_time_series, p, d, q, 0)

            # Calculate MAPE between actual and predicted values
            mape = calculate_mape(training_time_series, pred)
            
            # updating best loss if it is better
            if(mape < best_mape):
                best_mape = mape
                mapes.append(mape)
                best_p = p
                best_d = d
                best_q = q

            # Update weights using gradient descent
            p_gradient = np.zeros_like(p)
            q_gradient = np.zeros_like(q)

            for i in range(len(p)):
                p_temp = p.copy()
                p_temp[i] += 1e-6  # Small perturbation
                pred_temp = compute_arima(training_time_series, p_temp, d, q, 0)
                mape_temp = calculate_mape(training_time_series, pred_temp)
                p_gradient[i] = (mape_temp - mape) / 1e-6  # Approximate derivative

            for i in range(len(q)):
                q_temp = q.copy()
                q_temp[i] += 1e-6  # Small perturbation
                pred_temp = compute_arima(training_time_series, p, d, q_temp, 0)
                mape_temp = calculate_mape(training_time_series, pred_temp)
                q_gradient[i] = (mape_temp - mape) / 1e-6  # Approximate derivative

            # Update weights using gradient descent
            p -= learning_rate * p_gradient
            q -= learning_rate * q_gradient
        
    return best_p, best_d, best_q, mapes


# ######## #
# PART 1.1 #
# ######## #

def part_1_1():
    print("## ######## ##")
    print("## PART 1.1 ##")
    print("## ######## ##")
    print()
    
    # time series
    time_series = np.array([15., 10., 12., 17., 25., 23.])
    print("Time Series : ", time_series)
    print()

    # model co-parameters (student ID = 29970245 -> 0.2, 0.4, 0.5)
    p = [0.2, 0.4] # sig1, sig2, ...
    d = 1
    q = [0.5] # psi1, psi2, ...
    steps_ahead = 2
    print("Model Parameters :")
    print("\tp : ", len(p), " : ", p)
    print("\td : ", d)
    print("\tq : ", len(q), " : ", q)
    print("\tSteps Ahead : ", steps_ahead)
    print()
    
    # computing predictions
    print("Computing predictions...")
    pred = compute_arima(time_series, p, d, q, steps_ahead)
    print("Predictions computed.")
    print()
    
    # outputting predictions
    print("Predictions : ", pred)
    print("y_7 : ", pred[-2])
    print("y_8 : ", pred[-1])

# ######## #
# PART 1.2 #
# ######## #

def part_1_2():
    print("## ######## ##")
    print("## PART 1.2 ##")
    print("## ######## ##")
    print()
    
    # gathering time series
    ticker = "APPL"
    print("Testing with stock: ", ticker)
    print("Downloading stock data...")
    time_series = get_stock_prices(ticker)
    print("Stock data downloaded.")
    print()
    
    print("Creating training/testing split...")
    # creating training and testing data with an 80:20 split
    train_size = round(len(time_series) / 100  * 80)
    training_time_series = time_series[0:train_size]
    testing_time_series = time_series[train_size:len(time_series)]
    print("Training/testing split created.")
    print()
    
    # parameters for training
    p_range = [1,2,3]
    d_range = [1,2,3]
    q_range = [1,2,3]
    learning_rate = 0.001
    epochs = 100
    print("Training Parameters :")
    print("\tp range : ", p_range)
    print("\td range : ", d_range)
    print("\tq range: ", q_range)
    print("\tLearning rate: ", learning_rate)
    print("\tEpochs: ", epochs)
    print()
    
    
    # training arima model on training series
    print("Training ARIMA model...")
    print()
    p, d, q, mapes = train_arima(training_time_series, p_range, d_range, q_range, learning_rate, epochs)
    print()
    print("Model training complete.")
    print()
    print("Optimal p : ", len(p), " : ", p)
    print("Optimal d : ", d)
    print("Optimal q : ", len(q), " : ", q)
    print("Final testing accuracy : ", mapes[-1])
    print()
    
    # testing
    print("Testing model...")
    training_pred = compute_arima(training_time_series, p, d, q, 0)
    training_accuracy = calculate_mape(training_time_series, training_pred)
    testing_pred = compute_arima(testing_time_series, p, d, q, 0)
    testing_accuracy = calculate_mape(testing_time_series, testing_pred)
    print("Testing complete.")
    print("Testing accuracy : ", testing_accuracy)
    print()
    
    # plotting predictions
    fig, axes = plt.subplots(3, 1, figsize=(6,14))
    fig.tight_layout(pad=5.0)
    
    axes[0].set_title(f'{ticker}: Training Series Vs. Prediction \n MAPE={training_accuracy:.2f}%')
    axes[0].plot(training_time_series, label='True Series')
    axes[0].plot(training_pred, label='Prediction')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Price ($)")
    axes[0].set_xticklabels([])
    axes[0].set_xticks([])
    axes[0].legend()
    
    axes[1].set_title(f'{ticker}: MAPE Over Training Iterations')
    axes[1].plot(mapes, label='MAPE (%)')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].legend()
    
    axes[2].set_title(f'{ticker}: Testing Series Vs. Prediction \n MAPE={training_accuracy:.2f}%')
    axes[2].plot(testing_time_series, label='True Series')
    axes[2].plot(testing_pred, label='Prediction')
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Price ($)")
    axes[2].set_xticklabels([])
    axes[2].set_xticks([])
    axes[2].legend()

# #### #
# MAIN #
# #### #

if __name__ == "__main__":
    # please un-comment the part you would like to run

    part_1_1()

    #part_1_2()